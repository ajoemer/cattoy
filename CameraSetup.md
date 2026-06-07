Run this in windows powershell to attach the camera to wsl Ubuntu

List the busids
```bash
usbipd list
```

```bash
usbipd attach --busid 3-1 --wsl Ubuntu-Personal
```

Then back in WSL2:

sudo modprobe uvcvideo
ls /dev/video*
python detect_stream.py

In wsl Ubuntu, check cameras:
```bash
v4l2-ctl --list-devices
```