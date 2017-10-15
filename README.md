## Relativity Space - Internship Candidate Homework
This readme contains problems that candidates can choose to work on as a take-home assignment

### Instructions:
1. fork this repo
2. choose one of the problems from this repo to solve
    * You can use any language you want but python is recommended
    * Depending on your time and level of enthusiasm about the problem you can choose to do only parts of the problem
3. Once done send the link to your repo back to your interviewer

Open an issue on this repo if you have any questions about the problems.

Adding clarification and description as comments or readme file is welcomed if needed.

### Problem 1: Path Planning
An STL file (part.STL) is included in this repo. The 3D PDF (part.PDF) shows what this part looks like. Using this file, try to do as many of the below as possible:
	1. From the STL, create a point cloud representing the object.
	2. Using this point cloud as a set of possible points on the path of a 3D printer, find the optimized path (shortest in space) that traverses all the points you extracted.
	3. Requiring the printer motion to be monotonic in z, find the optimized path with this constraint.
	4. Assuming an additive process WITHOUT supporting material, try to develop a method of detecting features in the STL that will be difficult/impossible to print. Feel free to define conditions that are "difficult".
	5. Can you programatically determine an orientation of the part that reduces/eliminates these challenging features? 

### Problem 2 : Video Analysis
A video of a weld process is included in this repo (sample_weld_video.mp4). Programmatically, try to identify interesting features. Some examples would be:
	1. Identify the location of the weld pool in all frames
	2. Detect dropped frames/skipping events
	3. Detect welder on/off events
	4. Detect motion stopping
	5. Detect spatter
	Harder:
		6. Inclination of the weld path (in frame)
		7. Length of the wire between tip and weld pool
		8. Layer width estimation
		9. Any additional events/parameters/abnormalities