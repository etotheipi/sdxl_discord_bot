![](https://blogalanreiner.files.wordpress.com/2023/09/discord_bot_looking.jpg | width=500)

# Self-Hosted, Local-GPU SDXL Discord Bot

If you can run Stable Diffusion XL 1.0 (SDXL) locally using your GPU, you can use this repo to create a hosted instance as a Discord bot to share with friends and family.  

See the [related blog post](https://blog.alanreiner.com/2023/09/13/self-hosted-stablediffusionxl-discord-bot/).


Here's an
[animated .gif demo](https://blogalanreiner.files.wordpress.com/2023/09/sdxl_bot_demo-1.gif)
(this didn't work inline with Github Markdown)

## Features

* Default operation:
  * Tag the bot in a public channel or DM
  * The message after the tag will be used as the SDXL prompt
  * Generate 4 SDXL refined and responds with them as attachments
  * Takes about 60sec per 4 images on my NVIDIA RTX 3090
* Can include negative prompts and styles, separated by newlines
* Provides emoji feedback, changes status to indicate success or failure
* Proper logging to `logs/sdxl_bot.log`
* Threaded operation to provide simple responses during generation

## Installation

NOTE: Currently only tested on Linux + NVIDIA GPUs.  Other OSes will require some modification.

To install from scratch, you'll need to:

* Install NVIDIA Graphics Drivers
* Clone this repo: `git clone https://github.com/etotheipi/sdxl_discord_bot.git`
* Run the `download_and_setup.sh` script which will create the conda environment (including CUDA drivers) and download the SDXL model files
* Setup a new application in the [Discord developer portal](https://discord.com/developers/applications), give it message, reaction & attachment permissions
* Get the API token for the bot/application, and put it in your `~/.sdxlbot file`, which should be one line:  `token: $TOKEN` (without dollar sign)
* Invite it to your server [and any private channels]


## Running the bot:

```
$ ./run_sdxl_bot.sh [--cuda-device N]
```

...which is the same as...

```
$ conda activate sdxlbot   # conda env configured in initial script
$ python sdxl_bot.py [--cuda-device N]
```

Look at the bottom of `sdxl_bot.py` for command line arguments
You can check the `logs/` directory for runtime messages to help diagnose issues.

## Discussion & Recommendations

* As mentioned in the blog post, you could use Docker for this, but it gets hairy when GPUs and graphics drivers are involved.  It is certainly possible, but beyond the scope of this simple bot.
* Since this has to remain running all the time, it's recommended to run it as a background process.  My favorite way is via the `screen` tool.  This is a common utility, pre-bundled with many Linux OSes.  If not availble, should be easy to add via apt-get, yum, etc.
  * Type `screen`, which will open a new shell
  * Use the run script `./run_sdxl_bot.sh [--cuda-device N]` (which will do conda activate, and run the bot)
  * Type `Ctrl-A` (let go) then `d` to detach from the screen session
  * Use `screen -r` to reattach to view stdout or stop the process.  If you have multiple screen sessions, it will show you a list of availabe sessions, you can rerun via `screen -r PID` to attach to the correct one.
* Architecture: this bot uses [Py-Cord subclassing](https://guide.pycord.dev/popular-topics/subclassing-bots) which uses `asyncio` for main asynchronous processing loop.
  * I had difficulty invoking SDXL in a separate loop without blocking all bot processing, so I execute it in a separate `threading.Thread`
  * Mixing  `threading.Thread`s with asyncio meant that `run_sdxl()` could not invoke the asynchronous publishing of Discord messages.  Instead it puts the results on a queue and a separate async `task.loop(seconds=1)` will pick it up and publish.
* This currently only supports a single GPU with a single generation.  I intentionally did not allowing queuing requests, to avoid issues with massive queues requiring manual intervention to clear.  The bot will just say it's busy and to try again later.
  * However, it would be relatively easy (and safe) to expand support for multiple GPUs.  The main `on_message()` loop can track a queue/lock for each GPU and only reject if they are all full.  The output queue can be safely expanded to support near-simultaneous outputs.



Credit to [https://github.com/pydn/ComfyUI-to-Python-Extension][https://github.com/pydn/ComfyUI-to-Python-Extension] for the code to do the initial conversion from a working ComfyUI workflow to Python code.
