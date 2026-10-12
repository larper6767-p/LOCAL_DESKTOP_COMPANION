# Marin (desktop AI companion): test build

A voice AI that lives on your desktop as a Live2D character. 

# What you need
- Windows 10/11
- A microphone and speakers
- Your own free API keys: Cerebras (the brain) and Fish Audio (the voice)
- Internet (first run downloads the speech model, which is several GB)
- Ideally an NVIDIA GPU; without one it runs slower

# Setup (3 steps)
1. Double-click setup.bat, wait until it says done
2. Copy .env.example to .env and paste in your two keys, then double-click run.bat

# Using her
- talk to her (hold or press, then speak)
- Left-click her face: start a conversation
- Right-click: menu (size, liveliness, test actions, settings)
- System tray icon: quit

# Good to know
- Your speech is turned into text on your own computer
- The text of what you say goes to Cerebras, and what she says goes to Fish Audio
- Her memories and chat logs are saved only on your computer, in the config folder
- To wipe her memory, delete config/memory.json and config/episodes
- Please don't share the files or your keys with anyone

# If something is wrong
- Nothing shows, or she looks like a flat picture: run diagnose.bat
- No sound or no voice: check your Fish Audio key and your volume
- She doesn't hear you: check your default microphone in Windows

# Feedback 
- How long she takes to answer
- Anything that looks glitchy or feels robotic
- The three things you'd change first
