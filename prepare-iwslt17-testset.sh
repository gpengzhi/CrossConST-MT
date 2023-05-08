#!/usr/bin/env bash


SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
BPEROOT=subword-nmt/subword_nmt
BPE_CODE=iwslt17.tokenized/code

prep=iwslt17.tokenized.test
tmp=$prep/tmp
org=org

mkdir -p $org $prep $tmp


f=IWSLT17.TED.tst2017.mltlng.en-de.en.xml
tok=test.en-de.tok.en
grep '<seg id' $org/$f \
    | sed -e 's/<seg id="[0-9]*">\s*//g' \
    | sed -e 's/\s*<\/seg>\s*//g' \
    | sed -e "s/\’/\'/g" \
    | perl $TOKENIZER -threads 8 -l en > $tmp/$tok

f=IWSLT17.TED.tst2017.mltlng.de-en.de.xml
tok=test.en-de.tok.de
grep '<seg id' $org/$f \
    | sed -e 's/<seg id="[0-9]*">\s*//g' \
    | sed -e 's/\s*<\/seg>\s*//g' \
    | sed -e "s/\’/\'/g" \
    | perl $TOKENIZER -threads 8 -l de > $tmp/$tok

f=IWSLT17.TED.tst2017.mltlng.de-en.de.xml
ref=test.en-de.ref.de
grep '<seg id' $org/$f \
    | sed -e 's/<seg id="[0-9]*">\s*//g' \
    | sed -e 's/\s*<\/seg>\s*//g' \
    | sed -e "s/\’/\'/g" > $tmp/$ref


f=$tmp/test.en-de.tok.en
python3 $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > ${f/tok./bpe.}

f=$tmp/test.en-de.tok.de
python3 $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > ${f/tok./bpe.}


python3 add_tag.py --input $tmp/test.en-de.bpe.en --lang de


cat $tmp/test.en-de.tag.en > $prep/test.src
cat $tmp/test.en-de.bpe.de > $prep/test.tgt
