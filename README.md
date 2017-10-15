## Relativity Space - Candidate Homework 1
This readme contains problems that candidates can choose to work on as a take-home assignment

### Instructions:
1. Fork this repo
2. Choose at least one of the problems from this repo to solve
    * You can use any language you want but python is recommended
    * Depending on your time and level of enthusiasm about the problem you can choose to do only parts of the problem
3. Once done send the link to your repo back to your interviewer

Open an issue on this repo if you have any questions about the problems.

Adding clarification and description as comments or readme file is welcomed if needed.

### Problem 1: Path Planning
You are working on a path planning program for a 3D printer that prints using the FDM process. If you are not familiar with the FDM process read [its Wikipedia page](https://en.wikipedia.org/wiki/Fused_deposition_modeling). Given an STL file, a vector that indicates gravity, and maximum overhang angle that the printer can support, the goal is to find all vertices and edges from the STL file that require support to be printed. An STL file (part.STL) is included in this repo that can be used for testing and demonstrating your application.

1. STL file, maximum overhang angle, and gravity vector are inputs to your program
2. Assume build plate will be normal to gravity vector
3. Assume that your STL is always a shell structure (thin-walled)
3. To show your results, you can show all vertices and edges from STL file and differently color the ones that require support

### Problem 2 : Video Analysis
You are a control engineer for a robotic welder. You have a camera that records the welding process. You want to develop a program that detects certain features and events in your process which then allows you to trigger other controls required for the process. A sample video of a weld process is included in this repo (sample_weld_video.mp4) that can be used to test and demonstrate your program.

Your program is required to do following:
1. Identify the location of the weld pool in all frames
2. Detect dropped frames/skipping events
3. Detect welder on/off events
4. Detect motion stopping

Optional: If you feel like as super star you can work on any of the following as well:
1. Detect spatter
2. Inclination of the weld path (in frame)
3. Length of the wire between tip and weld pool
4. Angle between the wire and weld pool
5. Count pixels that can be used to estimate height of weld bead

