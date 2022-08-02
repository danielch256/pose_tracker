 Project diary
---

- Use a pose detector algorithm to make a proof of concept to  investigate the sport scientific value-add of using it on motor function testing.
- the goal is NOT to make the best pose detector.
- the goal is to investigate the idea of an easy to use realtime application solution on consumer hardware

this code should provide: rep counter, error tracker, x,y joint tracker, angle finder, speed and acceleration tracker
steps so far:
- pick a pose algo: openpose sucked to use and was slow as heck on my 1050ti laptop. blazepose runs well on cpu on windows, and was extremely easy to use.
- doing anything on a m1 mac was a nightmare at best.
- local code in pycharm on my old windows laptop, yay technology.
1. getting the pose algo to run: extremely simple
2. tracking (x,y) coordinates of different joints to use: pretty simple.
3. angle calculator: pretty simple.
4. rep counter and error tracker: difficult. why?
   1. defining the landing zones
   2. clearly defining a rep
5. landing zone finder approaches:
   1. define x,y boxes as a function of foot location (pass though y threshold = rep)
   2. find landing zones with some ml object detector: would have to be trained seperately
   3. find with cv approach: would have to be predefined color (easy, but needs specific landing plates)
   
   4. predefined mask: screenshot the video, pain mask.png, find those contours and draw bounding boxes. (bad, but good enough for now)
6. rep counter approaches:
   1. pose classifier (outside of scope)
   2. track v_x, v_y direction changes (pos/neg) --> smoothing this is more difficult to do in realtime than i'd anticipated, and outside my skillset rn
   24 fps is too low to get an accurate measurement and an interpolated vx and vy curve cant use be used to count reps, due to how interpolation works.
   3. a 120 fps input video would work better and be more accurate but the perfomance would presumably be significantly times worse.
   4. cv: if area around predefined pixel (starting point at frame 1) changes color = rep
   --> 4. cv: if (x,y) of a joint is within roi box change state to True, flip back to False when out etc. only works with predefined boxes
   
counter: by flipping in roi_1 or 2 from true to false and counting up on those = works. errors do nothing except say error (what else should they do.)
at this point id consider the functionality to be equal of a human, with caveats:
   1. only tested in on one video
   2. had to make a mask and load it to define roi1 and 2
   3. roi error detection works with bounding boxes of contours, not actual contours, because im a bad programmer.

2 and 3 are easy to fix with a board of 2 predefined colors for roi1 and 2, would be pixel accurate. or by training a model to recognize "board"

--> additional videos are probably going to be the big test

---
what's next? what can i actually do of value now?

- measure v and a of center of gravity, different joints etc.
- measure joint angles.
- measure time from landing to jump
- knee x-drift after landing