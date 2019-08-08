#!/bin/bash

REF_FILE=reference_output.dat
OUT_FILE=final_state.dat
EXE=d2q9-bgk.exe

./$EXE >& test.stdout

if diff $OUT_FILE $REF_FILE  ; then
    echo **TEST OK**
else
    echo **TEST FAILED**
fi

exit 0
