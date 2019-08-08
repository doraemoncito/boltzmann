#!/usr/bin/python

# 
# Script to write out obstacle coordinates
#

nrows = 200
ncols = 300

# Write out 'bottom rail'
for x in range(0, ncols):
    print x, "0", "1"

# Write out 'top rail'
for x in range(0, ncols):
    print x, "199", "1"

# Write out vertical bar
for y in range(60,121):
    print "99", y, "1"

