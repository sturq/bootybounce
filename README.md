# bootybounce

A desktop pet for Windows. A cut-out person stands on your taskbar and loops her animation in the background. Drag her somewhere, let go, and she drops back onto the taskbar.

![bootybounce on a Windows 11 desktop](docs/desktop.gif)

[Full quality video (MP4)](docs/desktop.mp4)

![Right-click menu](docs/screenshot.png)

## Use

Download `bootybounce-windows.zip` from [Releases](../../releases), unzip it and run `bootybounce.exe`. Keep the `pets` folder next to the exe.

- Left drag moves her. Release and she falls back onto the taskbar.
- Right click opens the menu: pick a person, her outfit, the size (Small, Medium, Large), always on top, quit.
- Clicks next to her go through to whatever is behind.
- Person, outfit, size, position and on-top are remembered in `%APPDATA%\Godot\app_userdata\bootybounce\settings.cfg`.

Mint is an AI-generated person.

## Add a person

Every person is a folder in `pets/`, with one folder per outfit:

```
pets/Person/Outfit/
  frames/0001.png ...   RGBA frames, full resolution
  hit.json              click polygon per frame
  pet.cfg               fps and pingpong=true|false
```

The program lists these folders, so a new person or outfit shows up in the menu without rebuilding.

`tools/make_pet.py` makes such a folder from a video. It pulls the frames, paints out a burned-in caption, cuts the person out with BEN2 through a ComfyUI that has the [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) nodes, picks a loop (a clean cut if the video has one, otherwise ping-pong between two still poses) and removes the old background colour from the edges. Ping-pong loops slow down to a stop before they turn, with in-between frames from optical flow, so the turn does not jerk.

```
pip install numpy opencv-python-headless imageio-ffmpeg
python3 tools/make_pet.py video.mp4 Person/Outfit --comfy http://<comfyui-host>:8188 [--caption Y0,Y1] [--box X0,Y0,X1,Y1] [--loop A,B]
```

Mint/Original was made with `--caption 380,540 --box 100,380,620,1080 --loop 0,186`: the whole clip forward and back, 13 seconds.

## Develop

Godot 4.7, compatibility renderer, one script (`pet.gd`).

- `godot --headless --path . --script res://test.gd` checks every pet (frames, click polygons, no jump in the loop).
- `bootybounce.exe -- --shot=out.png --frame=N [--pet=Name] [--size=Large]` saves what the window shows, with alpha, prints its state as JSON and quits.
- A tag `v*` makes GitHub Actions run the test, export the Windows build and attach the zip to a release.
