from datetime import datetime
from copy import deepcopy
import argparse
import asyncio
import json
import numpy as np
import os
import queue
import random
import requests
import sys
import threading
import tiktoken
import time
import yaml

from PIL import Image
from io import BytesIO
import discord
from discord.ext import commands, tasks

    

################################################################################
### LOGGING SETUP
import logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()

os.makedirs('./logs', exist_ok=True)
fileHandler = logging.FileHandler(os.path.join('logs', f'sdxl_bot.log'))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
################################################################################



################################################################################
# Using a global queue.Queue to collect bot outputs from SDXL threads, rather than
# trying to invoke async methods from a threading.Thread instance.  A separate
# bot process/loop will wait for these responses and auto publish to the server
# This should be the only global
global response_queue
response_queue = queue.Queue(maxsize=1)

                
class SdxlBot(discord.Bot):
    def __init__(self, args):
        
        # Hardcoding intents for this specific bot
        intents = discord.Intents.default()
        intents.messages = True
        intents.members = True
        
        # Priveleged intents not necessary off-the-shelf, but maybe if you add functionality...
        #intents.typing = False
        #intents.presences = False
        #intents.message_content = True
            
        # Normal initialization for discord.Bot
        super().__init__(command_prefix='!', intents=intents)
        
        discord_key_file = yaml.safe_load(open(os.path.expanduser(args.discord_key_file), 'r'))
        self.discord_token = discord_key_file['token']
        self.styles_list = json.load(open('StyleSelectorXL/sdxl_styles.json', 'r'))
        self.styles_by_name_lower = {s['name'].lower():s for s in self.styles_list}
        self.sdxl_lock = threading.Lock()
        self.curr_sdxl_thread = None
        self.response_queue = queue.Queue(maxsize=1)

        ################################################################################
        # Loading the models upfront into globals so that it persists across threads 
        # and async bot operations.  Subclassing the discord bot and including these
        # directly would've been a better strategy.
        base_ckpt='sd_xl_base_1.0.safetensors'
        ckpt_loader = CheckpointLoaderSimple()
        self.ckpt_base = ckpt_loader.load_checkpoint(ckpt_name=base_ckpt)

        refiner_ckpt='sd_xl_refiner_1.0.safetensors'
        ckpt_loader = CheckpointLoaderSimple()
        self.ckpt_refiner = ckpt_loader.load_checkpoint(ckpt_name=refiner_ckpt)
        
        logger.info('Styles loaded:')
        logger.info(self.get_formatted_style_list(ncols=3))
        
        self.auto_message_sender_loop.start()
        
    def run(self, token=None):
        if token is None:
            token = self.discord_token
            
        super().run(token)
        
        
    def get_formatted_style_list(self, ncols=3, col_width_chars=25):
        style_names = sorted([s['name'] for s in self.styles_list])
        nstyles = len(style_names)
        nrows = int(nstyles/ncols) + 1
        row_list = []
        for r in range(nrows):
            row = ''
            for c in range(ncols):
                idx = r + nrows*c
                if idx < nstyles:
                    row += style_names[idx].ljust(col_width_chars)
            row_list.append(row)
        return '\n'.join(row_list)

    
    async def on_ready(self):
        logger.info(f'SdxlBot has connected to Discord!')


    async def get_bot_role_id(self, message):
        if message.guild is None:
            return None

        bot_member = message.guild.get_member(self.user.id)
        for role in bot_member.roles:
            if role.name == self.user.name:
                return role.id

        return None

    
    async def on_message(self, message):

        if isinstance(message.channel, discord.DMChannel):
            is_dm = True
            channel_name = f'DM({message.channel.recipient})'
        else:
            is_dm = False
            channel_name = message.channel.name

        logger.info(f"Message Author (ID): {message.author.name} ({message.author.id})")
        logger.info(f"Message Channel (ID): {channel_name} ({message.channel.id})")
        logger.info(f"Message Content (ID): {message.content} \n(Message ID: {message.id})")
        logger.info(f"Message Mentions: {message.mentions}")
        logger.info(f"Message Embeds: {message.embeds}")
        logger.info(f"Message Attachments: {message.attachments}")

        role_id = await self.get_bot_role_id(message)
        logger.info(f"Bot's User Name (ID) [RoleID]: {self.user.name} ({self.user.id}) [{role_id}]")

        is_mentioned = False
        if self.user.name in [m.name for m in message.mentions]:
            is_mentioned = True

        user_message = message.content
        mention_search = [f'<@{self.user.id}>', f'<@!{self.user.id}>', f'<@&{role_id}>']
        for m in mention_search:
            # Search for mentions, and also remove them if they are there
            user_message = user_message.replace(m, "").strip()
            if m in message.content:
                is_mentioned = True

        is_mentioned = is_mentioned or is_dm

        if message.author.id == self.user.id or not is_mentioned:
            return

        await message.add_reaction(u"\U0001f916") # robotface

        if is_dm and args.disable_dm:
            await message.add_reaction(u'\u274c')
            err_msg = f'Sorry, DMs are not authorized.  ' 
            await message.channel.send(err_msg)
            #await self.process_commands(message)

        prompt_lines = [l.strip() for l in user_message.split('\n') if len(l.strip()) > 0]
        logging.info("prompt_lines:")
        for l in prompt_lines:
            logging.info('  ' + l)

        # First line is always expected to be the positive prompt, negative and style optional
        pos = prompt_lines[0]
        neg = ''
        style_name = None

        # Remove if they accidentally put "Positive:" or "Positive prompt:" in their prompt
        if pos.lower().startswith('positive'):
            brk_index = pos.find(':')
            pos = pos[brk_index+1:].strip()

        # Check any other lines for 
        for line in prompt_lines[1:]:
            if len(line) > 1:
                if line.lower().startswith('negative'):
                    neg = line[line.find(':')+1:].strip()
                elif line.lower().startswith('style'):
                    style_name = line[line.find(':')+1:].strip()
                else:
                    # I guess this is part of the positive prompt!?
                    pos += ' ' + line

        # If they picked a style, make sure it exists
        if style_name is not None:
            if style_name.lower() not in [s['name'].lower() for s in self.styles_list]:
                err_msg = f'Style "{style_name}" not found. Type "liststyles" to see all options.'
                await message.channel.send(err_msg)
                await message.add_reaction(u'\u274c')
                return

            style_info = self.styles_by_name_lower[style_name.lower()]
            pos = style_info['prompt'].replace('{prompt}', pos.strip().strip('.'))

            if len(neg) == 0:
                neg = style_info['negative_prompt']
            else:
                neg = neg.strip().strip(',') + ', ' + style_info['negative_prompt']


        # User requests help
        bot_help_msg = 'If you want me to generate an SDXL image, tag me and provide the positive prompt ' \
                       'as the body of the message.  If you want to include a negative prompts, simply ' \
                       'add two newlines (shift-enter) to the end, and then add "Negative: <prompt>"'

        if user_message.startswith('HELP'):
            await message.channel.send(bot_help_msg)
            #await self.process_commands(message)
            return

        if user_message.startswith('CANCEL'):
            # I need to do something to curr_sdxl_thread, though I hear this may be dangerous...?
            return #???

        # User requests list of styles.  Print it in three columns
        if any([user_message.startswith(s) for s in ['styles', 'liststyles', 'stylelist', 'list styles']]):
            style_list_str = self.get_formatted_style_list()
            await message.channel.send(f'```{style_list_str}```')
            #await self.process_commands(message)
            return

        if self.sdxl_lock.locked():
            await message.add_reaction(u'\u274c') 
            await message.channel.send( f'{message.author.mention} There is another ' \
                                         'image generation in process.  Try again later.')
            #await self.process_commands(message)
            return
        
        img2img_path = None
        if len(message.attachments) > 0:
            attachment = message.attachments[0]
            if attachment.filename.lower().endswith(('png', 'jpg', 'jpeg')):
                # Comfy doesn't seem to give me a choice about this.  LoadImage expects it here
                save_dir = './ComfyUI/input'  
                os.makedirs(save_dir, exist_ok=True)
                
                response = requests.get(attachment.url)
                attached_img = Image.open(BytesIO(response.content))
                attached_img = attached_img.convert('RGB')
                
                time_string = datetime.now().strftime('%Y_%m_%d_%H%M')
                orig_fn = f'{time_string}_img2img_orig.jpg'
                orig_path = os.path.join(save_dir, orig_fn)
                attached_img.save(orig_path)
                img2img_path = self.crop_resize_save_image(orig_path)
                # Comfy's LoadImage expects just the filename, and will look in ComfyUi/input for it
                img2img_path = os.path.basename(img2img_path)
                
                

        try:
            sdxl_request_params = {
                'requestor': message.author.mention,
                'positive_prompt': pos,
                'negative_prompt': neg,
                'message_obj': message
            }

            sdxl_kwargs = {
                'req_map': sdxl_request_params,
                'include_base_output': args.include_base_output,
                'batch_size': args.gen_count,
                'num_steps_base': args.base_steps,
                'denoise_base': args.base_denoise,
                'num_steps_refiner': args.refiner_steps,
                'denoise_refiner': args.refiner_denoise,
            }
            
            if img2img_path is None:
                # Regular, text-only input SDXL
                self.curr_sdxl_thread = threading.Thread(target=self.run_sdxl, kwargs=sdxl_kwargs)
                self.curr_sdxl_thread.start()
            else:
                # Img2Img SDXL
                sdxl_request_params['input_image_path'] = img2img_path
                sdxl_kwargs = {
                    'req_map': sdxl_request_params,
                    'num_steps_base': args.base_steps,
                    'denoise_base_img2img': args.base_denoise_img2img,
                }
                self.curr_sdxl_thread = threading.Thread(target=self.run_sdxl_img2img, kwargs=sdxl_kwargs)
                self.curr_sdxl_thread.start()
                await message.add_reaction(u"\U0001F5BC") # indicate img2img

            await self.change_presence(status=discord.Status.dnd, activity=discord.Game(name="GENERATING..."))
            await message.add_reaction(u"\U0001F44D") # thumbsup


        except asyncio.CancelledError:
            await message.channel.send(f'Sorry {message.author.mention}, your request was canceled')
            logger.info('Task was canceled')
            return
        
        
    ########
    # This was adapted from the "ComfyUI-to-Python-Extension" project on github.  I used it to convert
    # my ComfyUI workflow to pure python code.  I tried using HuggingFace and code straight from
    # StabilityAI repo, but they both killed my VRAM while ComfyUI somehow figured it out.
    # If I figure out what went wrong, I'd love to switch back to plain HuggingFace.
    def run_sdxl( self,
                  req_map,

                  img_width=1024,
                  img_height=1024,
                  batch_size=4,

                  seed_base=None,
                  num_steps_base=25,
                  denoise_base=1.0,

                  seed_refiner=None,
                  num_steps_refiner=10,
                  denoise_refiner=0.25,

                  include_base_output=False,
        ):

        img_output_dir = 'ComfyUI/output'  # ComfyUI doesn't want to let me change this...
        os.makedirs(img_output_dir, exist_ok=True)

        with self.sdxl_lock:
            logging.info('SDXL Thread Started')
            # This shouldn't be needed, but in case something goes wrong, limit it to 10Hz
            time.sleep(0.1)

            # This is a dictionary with the keys "requestor", "positive_prompt", "negative_prompt", "message_obj"
            # We will add "fn_list_base" and "fn_list_refined"
            requestor = req_map['requestor']
            positive_prompt = req_map['positive_prompt']
            negative_prompt = req_map['negative_prompt']

            logger.info('Request received in run_sdxl()')
            logger.info(f'Requestor: """{requestor}"""')
            logger.info(f'Positive prompt: """{positive_prompt}"""')
            logger.info(f'Negative prompt: """{negative_prompt}"""')


            seed_base    = random.getrandbits(64) if seed_base    is None else seed_base
            seed_refiner = random.getrandbits(64) if seed_refiner is None else seed_refiner

            with torch.inference_mode():

                cliptextencode = CLIPTextEncode()
                clip_base_encode_pos  = cliptextencode.encode(text=positive_prompt, clip=self.ckpt_base[1])
                clip_base_encode_neg  = cliptextencode.encode(text=negative_prompt, clip=self.ckpt_base[1])

                clip_refiner_encode_pos = cliptextencode.encode(text=positive_prompt, clip=self.ckpt_refiner[1])
                clip_refiner_encode_neg = cliptextencode.encode(text=negative_prompt, clip=self.ckpt_refiner[1])

                eli = EmptyLatentImage()
                emptylatentimage = eli.generate(width=img_width, height=img_height, batch_size=batch_size)

                ksampler = KSampler()
                vaedecode = VAEDecode()
                saveimage = SaveImage()

                ksampler_base = ksampler.sample(
                    seed=seed_base,
                    steps=num_steps_base,
                    cfg=6,
                    sampler_name="dpmpp_2s_ancestral",
                    scheduler="normal",
                    denoise=denoise_base,
                    model=self.ckpt_base[0],
                    positive=clip_base_encode_pos[0],
                    negative=clip_base_encode_neg[0],
                    latent_image=emptylatentimage[0],
                )

                if not include_base_output:
                    logger.info('Skipping base image decoding')
                    req_map['fn_list_base'] = []
                else:
                    logger.info('Decoding base image for output')
                    vae_base_decode = vaedecode.decode(samples=ksampler_base[0], vae=self.ckpt_base[2])
                    base_out = saveimage.save_images(filename_prefix="unrefined_output", images=vae_base_decode[0])
                    req_map['fn_list_base'] = [os.path.join(img_output_dir, fn['filename']) for fn in base_out['ui']['images']]

                ksampler_refiner = ksampler.sample(
                    seed=seed_refiner,
                    steps=num_steps_refiner,
                    cfg=8,
                    sampler_name="dpmpp_2m",
                    scheduler="normal",
                    denoise=denoise_refiner,
                    model=self.ckpt_refiner[0],
                    positive=clip_refiner_encode_pos[0],
                    negative=clip_refiner_encode_neg[0],
                    latent_image=ksampler_base[0],
                )

                vae_refiner_decode = vaedecode.decode(samples=ksampler_refiner[0], vae=self.ckpt_refiner[2])
                refiner_out = saveimage.save_images(filename_prefix="sdxl_output", images=vae_refiner_decode[0])

                req_map['fn_list_refined'] = [os.path.join(img_output_dir, fn['filename']) for fn in refiner_out['ui']['images']]
                self.response_queue.put(req_map)

    def run_sdxl_img2img(self,
                  req_map,
                  seed_base=None,
                  num_steps_base=20,
                  denoise_base_img2img=0.4,
        ):

        img_input_dir = 'ComfyUI/input'  # ComfyUI doesn't want to let me change this...
        img_output_dir = 'ComfyUI/output'  # ComfyUI doesn't want to let me change this...
        os.makedirs(img_output_dir, exist_ok=True)

        with self.sdxl_lock:
            logging.info('SDXL Thread Started')
            # This shouldn't be needed, but in case something goes wrong, limit it to 10Hz
            time.sleep(0.1)

            # This is a dictionary with the keys "requestor", "positive_prompt", "negative_prompt", "message_obj"
            # We will add "fn_list_base" and "fn_list_refined"
            requestor = req_map['requestor']
            positive_prompt = req_map['positive_prompt']
            negative_prompt = req_map['negative_prompt']
            input_image_path = req_map['input_image_path']

            logger.info('Request received in run_sdxl()')
            logger.info(f'Requestor: """{requestor}"""')
            logger.info(f'Positive prompt: """{positive_prompt}"""')
            logger.info(f'Negative prompt: """{negative_prompt}"""')
            logger.info(f'Input Image: """{input_image_path}"""')

            seed_base    = random.getrandbits(64) if seed_base    is None else seed_base

            with torch.inference_mode():

                cliptextencode = CLIPTextEncode()
                clip_base_encode_pos  = cliptextencode.encode(text=positive_prompt, clip=self.ckpt_base[1])
                clip_base_encode_neg  = cliptextencode.encode(text=negative_prompt, clip=self.ckpt_base[1])

                loadimage = LoadImage()
                load_image = loadimage.load_image(image=input_image_path)

                vae_base_encode = VAEEncode()
                img2img_encoded = vae_base_encode.encode(pixels=load_image[0], vae=self.ckpt_base[2])

                ksampler = KSampler()
                vaedecode = VAEDecode()
                saveimage = SaveImage()

                ksampler_base = ksampler.sample(
                    seed=seed_base,
                    steps=num_steps_base,
                    cfg=6,
                    sampler_name="dpmpp_2s_ancestral",
                    scheduler="normal",
                    denoise=denoise_base_img2img,
                    model=self.ckpt_base[0],
                    positive=clip_base_encode_pos[0],
                    negative=clip_base_encode_neg[0],
                    latent_image=img2img_encoded[0],
                )

                logger.info('Decoding base image for output')
                vae_base_decode = vaedecode.decode(samples=ksampler_base[0], vae=self.ckpt_base[2])
                base_out = saveimage.save_images(filename_prefix="sdxl_img2img_output", images=vae_base_decode[0])
                print(yaml.dump(base_out, indent=2))
                req_map['fn_list_base'] = [os.path.join(img_input_dir, input_image_path)]
                req_map['fn_list_base'].append(os.path.join(img_output_dir, base_out['ui']['images'][0]['filename']))
                req_map['fn_list_refined'] = []
                self.response_queue.put(req_map)
                
                
    def crop_resize_save_image(self, orig_image_path):
        if not os.path.exists(orig_image_path):
            return None
        
        img = np.array(Image.open(orig_image_path))
        w,h,c = img.shape
        print('Input image shape:',w,h,c)
        if w > h:
            img = img[w//2-h//2:w//2+h//2, :, :]
        if h > w:
            img = img[:, h//2-w//2:h//2+w//2, :]
            
        print('Output image shape:', img.shape)
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img = img.resize((1024, 1024))
        
        time_string = datetime.now().strftime('%Y_%m_%d_%H%M')
        output_path = orig_image_path[:-9] + '_cropped.jpg'
        output_path = '_'.join(orig_image_path.split('_')[:-1]) + '_cropped.jpg'
        img.save(output_path)
        return output_path
        
                    
    @tasks.loop(seconds=1)
    async def auto_message_sender_loop(self, out_map=None):
        if out_map is None:
            try:
                out_map = self.response_queue.get_nowait()
            except queue.Empty:
                return
        else:
            logger.info('auto_message_sender_loop() called directly...?')

        logger.info('Auto-send received something:')
        debug_map = {k:v for k,v in out_map.items() if k != 'message_obj'}
        logger.info('\n'+json.dumps(debug_map, indent=2))
        message = out_map['message_obj']
        pos = out_map['positive_prompt']
        neg = out_map['negative_prompt']
        response_text = f"{message.author.mention}\nPositive prompt: {pos}\n\nNegative prompt: {neg}"

        fn_list = out_map['fn_list_base'] + out_map['fn_list_refined']
        await self.change_presence(status=discord.Status.online)
        await message.channel.send(response_text, files=[discord.File(f) for f in fn_list])



if __name__ == '__main__':
    
    instructions = """
    To use this script make sure to do the following:
    1. Put discord bot info in ~/.sdxlbot with the following yaml structure:
          token: "..."
    2. Make sure you have activated the environment with the packages from requirements.txt installed
    3. Make sure model files (base and refiner) are in the ComfyUI/models/checkpoints directory with std names
    3. python sdxl_bot.py [args]
    """
    
    parser = argparse.ArgumentParser(
                    prog='sdxl_bot.py',
                    description='Run a Discord Bot that serves SDXL from your local GPU',
                    epilog=instructions)
                                            
    parser.add_argument('--cuda-device', type=str, default=None, help='Index of GPU to use, 0-N')
    parser.add_argument('--include-base-output', action='store_true', help="Include unrefined intermediate output images")
    parser.add_argument('--discord-key-file', type=str, default='~/.sdxlbot', help='Path to file containing discord bot token')
    parser.add_argument('--gen-count', type=int, default=4, help='Number of images to produce for each prompt. Default is 4 (8 if including base output).')
    parser.add_argument('--disable-dm', action='store_true', help='Do not allow direct messages to bot.')
    parser.add_argument('--base-steps', type=int, default=25, help='Number of steps to use for the base model')
    parser.add_argument('--base-denoise', type=float, default=1.0, help='Denoise strength of base model')
    parser.add_argument('--base-denoise-img2img', type=float, default=0.5, help='Denoise strength of base model for img2img (refiner skipped)')
    parser.add_argument('--refiner-steps', type=int, default=10, help='Number of steps to use for the refiner model')
    parser.add_argument('--refiner-denoise', type=float, default=0.25, help='Denoise strength of refiner model')
    parser.add_argument('--debug', action='store_true', help='Include debug logging output')
    args = parser.parse_args()
    
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
                        
    if args.cuda_device is not None:
        logger.info('CUDA_VISIBLE_DEVICES set to "' + str(args.cuda_device) + '"')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
                        

    # These imports needed to happen after setting CUDA_VISIBLE_DEVICES above, can't
    # put them at the top of the file.  I'm sure there's another way to do this...
    import torch
    
    sys.path.append('./ComfyUI')
    from nodes import (
        EmptyLatentImage,
        CheckpointLoaderSimple,
        VAEDecode,
        VAEEncode,
        CLIPTextEncode,
        KSampler,
        SaveImage,
        LoadImage,
    )

    
    sdxl_bot = SdxlBot(args)                     
    sdxl_bot.run()
    
                        
                        





