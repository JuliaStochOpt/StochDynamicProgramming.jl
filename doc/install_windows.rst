.. _install_windows:

===========================================
Install StochDynamicProgramming on Windows
===========================================


Installing Julia
~~~~~~~~~~~~~~~~

StochDynamicProgramming is a Julia package so if you want to use it, you
need first to install Julia depending on whether your computer is 32-bit
or 64-bit. To determine that, follow the steps
`here <http://windows.microsoft.com/fr-fr/windows7/find-out-32-or-64-bit>`__.

Once it is done, you can follow this
`link <http://julialang.org/downloads/>`__ to install Julia (command
line version) appropriate to your computer.

Installing the package in Julia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the Julia terminal using the desktop icon or looking for it in the
search bar of Windows. Then copy-past or enter the following line::

    julia> Pkg.add("StochDynamicProgramming.jl")

Once it is done, you can use the package. To see where the files are located,
you can enter the command::

    julia> pwd()

in the command line of Julia.

Compiling the doc
=================

To compile the doc, you will need a C compiler to execute a makefile,
Python and Sphinx installed.

Installing Python and Sphinx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow this
`link <http://www.sphinx-doc.org/en/stable/install.html#windows-install-python-and-sphinx>`__
to install Python and choose a version x86 if you want a 32-bits version
and choose x86-64 if you want a 64 bits version. Be careful to add the
PATH variable as it is explained.

Installing MinGW
~~~~~~~~~~~~~~~~

Just follow the steps explained
`here <http://www.mingw.org/wiki/Getting_Started>`__.

Compiling the doc
~~~~~~~~~~~~~~~~~

Once you get Python, Sphinx and MinGW, open a command line. Go to
StochDynamicProgramming folder. Enter::

    cd doc

and then::

    make html

Have a nice use
===============

| Now everything is installed, you can enjoy the use of the package. If
you want some example, we invite you to try our Julia Notebooks available
`here <https://github.com/leclere/StochDP-notebooks/blob/master/notebooks/damsvalley.ipynb>`__.
 
| Don't hesitate to contact us if you encounter any problem to let us improve the
document for further users !!
