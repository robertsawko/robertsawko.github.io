---
layout: post
title: "It is alive!"
date: 2015-04-30 20:41:25 +0100
comments: true
categories: 
---

This is my first post using [octopress](http://octopress.org/) i.e. a git based
blogging engine. At the moment I am just testing various functionalities. Is
markdown **bold** working? Do we have MathJax?

## The quest for MathJax

Let's try the Boltzmann equation:

$$
\begin{equation}
\frac{\partial f}{\partial t} 
+ \frac{\mathbf{p}}{m} \cdot \nabla f
+ \mathbf{F} \cdot \frac{\partial f}{\partial \mathbf{p}}
=
{\left(\frac{\partial f}{\partial t} \right)}_{\mathrm{collision}}
\end{equation}.
$$

To test inline maths we will use $x^2+y^2=z^2$ where $x$, $y$ are catheti
and $z$ is a hypotenuse of a right triangle.

It works! Although to be truthful it did not work out-of-the-box. I had to
fight a small battle messing around with some files. At the moment I am using
`kramdown` and CDN according to advice given on this
[post](http://www.idryman.org/blog/2012/03/10/writing-math-equations-on-octopress/).
The only difference is that my *javascript* code landed in

`source/_includes/custom/head.html`

as it is being sourced by the main head anyway. This looks like a slightly
cleaner solution.

But that wasn't enough! After uploading the files on Github Pages they wouldn't
display even though I could see them in the preview mode. The reason was the
HTTP secure access to MathJax. It's important that CDN address contains `https`
rather than `http`. There's a
[passage](http://docs.mathjax.org/en/latest/start.html#secure-cdn-access) in
MathJax documentation about it. So much for "setting up a scientific blog in
half-an-hour", but I've learnt a few things definitely.

MathJax test passed.

## The plan...

... is to run this as a little experiment in curiosity, amusement and memory of
all these little intellectual pursuits which somehow get lost in the daily
routine of existence. I would like to focus here on the work I do related to
fluid dynamics and coding but other content may appear too. The first few steps
though I need make is to investigate:

 * adding sub-pages,
 * add contact info,
 * test iPython integration,
 * check comment functionalities (in the unlikely case someone wants to
   comment!),
 * theme customization.

Interestingly at this stage the lists were **not** indented properly. Again, I
found the
[instructions](http://stackoverflow.com/questions/24794024/markdown-list-does-not-indent-using-octopress)
which fixed it though. During the whole process I was struck by the fact that I
am overwhelmed by the technological nomenclature. There are lots of concepts
here I completely don't understand or just heard about for the very first time.
Hopefully this will become less of an impediment as time goes on.

The first actual post to appear is going to be on the introduction to Kraichnan
theory of turbulence which I am reading about at the moment. Will try to
reproduce some of the results and test iPython integration. Future posts may
cover some adventures in population balance modelling, partial differential
with stochastic inputs and fluid dynamics problems.
