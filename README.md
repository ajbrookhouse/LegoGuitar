# LegoGuitar
Script with functions useful for figuring out how to build the shape of a guitar with legos and decorate it based on an image

This is a script that has been assisting me on a project that I am working on this summer. I am making a guitar out of legos. To figure out how to get the shape right, I decided that using a python program would be helpful.

The guitar is going to be made by creating a bottom from two layers of Lego Plates that are staggered to make a bigger plate in the shape of a guitar. The sides and part of the middle will be built higher with lego bricks and the top will be covered by two layers of plates like the bottom. The top face then will be covered with small Lego Tiles to make it smooth and to decorate the top. Parts of the inside I will pour epoxy into to give the guitar body strength and provide a thick area for screws to grab onto, specifically for the neck and bridge.

The two main uses of this script are to figure out the shape of a guitar, quantized to lego dimensions, and to take an image and make a mosaic out of lego tiles.

If you just run the script as is, it will plot the Les Paul guitar shape, shown a few different ways with matplotlib. If you are interested in how it creates images on the guitar, uncomment line 205 and provide an image path to try.

If you want to make other guitar shapes, you will need to adjust the parameters for the guitar equation used in the guitarCurve function. I found this equation and parameters on this website (http://www.mnealon.eosc.edu/The_Nealon_Equation.html). It has the equation parameters for other shapes like Strat and Tele.

As I progress with the project, I will upload images to show how it worked out! I just got the last legos I needed to get started and am now working on building the bottom plate. If anyone else tries something like this or uses this program, let me know! I would think that is really cool.

Here is some example output the program produced. The first just shows how to make the shape of the guitar, and the second is turning an image I found online into a lego mosaic on the guitar.

Images/GuitarGridAndLines.png

Images/Stormtrooper_Guitar.png
