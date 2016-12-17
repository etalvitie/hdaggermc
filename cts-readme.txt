**************************  
  Context Tree Switching
**************************

Author:    Joel Veness
Date:      14/11/2011

Introduction:
-------------

This is an implementation of the Context Tree Switching algorithm, 
described in the accompanying technical report at 
    http://arxiv.org/abs/1111.3182v1. 

It is an extension of the Context Tree Weighting algorithm, and works by replacing
the recursive weighting step by a computationally efficienct switching 
technique. The probabilities estimated by this technique then drive a 
standard binary arithmetic encoder, which produces the compressed file.

This is a proof-of-concept implementation, and is currently quite memory hungry. 
However, provided you have a modern machine with 2 gigs or more of RAM, the default
settings should work fine for files less than 10meg.

Program Usage:
---------------

Run cts.exe --help for program options. It should display something like:

Usage:
  --help
  --depth arg (=48)      Maximum depth to look back in bits. Higher values use more RAM.
  --method arg (=faccts) Compression method. (ctw, cts, facctw, faccts)
  --file arg             File to compress/decompress. A compressed file with the same name plus an
                         extension (e.g. foo.cts) will be produced. Decompression is chosen
                         automatically if the file extension is .ctw, .cts, .facctw, or .faccts.

A sample usage might go like this:

> cts.exe --depth 128 --file foo

Later, to recover foo, run

> cts.exe --depth 128 --file foo.faccts

About the different methods:
----------------------------

ctw:	A bare-bones implementation of vanilla CTW, with no enhancements.
cts:    An implementation of cts. Treats everything as a raw binary stream.
facctw: CTW plus a binary decomposition to exploit byte oriented data.
faccts: CTS plus a binary decomposition to exploit byte oriented data

There are some optional flags that can be set in the source code.
Much better CTW performance can be obtained by enabling the zero-redundancy 
estimator and disabling weighting at byte boundaries. For the best CTW
implementation I am aware of, see http://www.ele.tue.nl/ctw/.

The most important parameter in CTS seems to be the choice of prior weight to assign
to a switch (controlled by the log_switch_prior and log_kt_prior parameters in the code). 
The analysis and results reported in the technical report for vanilla CTS use a value of 0.5. 
Much better across the board performance appears when log_switch_prior is set to 0.925.
The code here uses this better performing constant as the default.

Additional Credits:
--------------------

In addition to my co-authors on the attached technical report, I'd like to mention:

   - Frans Willems, Yuri Shtarkov, Tjalling Tjalkens, and Paul Volf for their fantastic papers!
     Their work got me interested in data compression, and in particular techniques that mix theory and practice.
     
   - The CTW 0.1 website (http://www.ele.tue.nl/ctw/) contained lots of helpful ideas.
     In particular, the "Strict Unique Path Pruning" idea can luckily be adapted to CTS,
     which kept the memory requirements to a much more reasonable level.
     
Musings/Todo:
--------------
   
   - The settings used to compress and decompress need to match. If they don't, all hell will break loose.
     In the future this should be done properly by storing appropriate information in the file header.
     
   - The memory requirements can be reduced considerably by packing the information
     required at each node. This would be good to do in the future.
     
   - This implementation could be improved by converting the logic to use only integer arithmetic.
   
   - There are probably many ways to speed this implementation up considerably. 
     I'd welcome some suggestions, since this is my first data compression program.
     
   - It's quite likely that compression performance can be improved by tuning some parameters inside cts.cpp. 
     Changing the global log_kt_prior constant to a depth dependent constant should be investigated.
   
   - For maxmimum compression, disable the fast math routines in cts.cpp. By default these settings
     are enabled since the gain in speed is large. This more than offsets the loss in compression,
     which I found to be within a margin of ~= 0.01 bits per character on the Calgary Corpus.
     
     