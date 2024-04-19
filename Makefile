# Glorified file to write down commands for things that are relevant and may be important to a pipeline later

kill-b-frames:
	ffmpeg -i out.mp4 -x264opts bframes=0:keyint_min=250 vid_p_i_only.mp4

freeze-i-frames:
	ffmpeg -skip_frame nokey -i vid_p_i_only.mp4 -vsync 0 -frame_pts true frames/i/frame_%d.png 

freeze-frames:
	ffmpeg -i vid_p_i_only.mp4 -vsync 0 -frame_pts true frames/p/frame_%d.png 

# To delete the i frames from the `p` folder (so we actually end up with only p frames) run the `delete_is.py` script from the `frames` folder.

activate:
	source ~/anaconda3/bin/activate