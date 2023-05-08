#!/usr/bin/env bash

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git


SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000

prep=iwslt17.tokenized
tmp=$prep/tmp
org=org

mkdir -p $org $prep $tmp


tar zxvf 2017-01-trnmted.tgz
tar zxvf 2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz
mv DeEnItNlRo-DeEnItNlRo/train.* $org/
mv DeEnItNlRo-DeEnItNlRo/*.xml $org/
rm -rf DeEnItNlRo-DeEnItNlRo/
rm -rf 2017-01-trnmted/


tar zxvf 2017-01-mted-test.tgz
for SRC in "de" "en" "it" "nl" "ro"; do
    for TGT in "de" "en" "it" "nl" "ro"; do
        if [ $SRC != $TGT ]; then
            tar zxvf 2017-01-mted-test/texts/$SRC/$TGT/$SRC-$TGT.tgz
            mv $SRC-$TGT/*.xml $org/
            rm -rf $SRC-$TGT/
        fi
    done
done
rm -rf 2017-01-mted-test/


echo "pre-processing train data..."
for l in "de" "it" "nl" "ro"; do
    f=train.tags.en-$l.en
    tok=train.en-$l.tok.en
    cat $org/$f \
        | grep -v '<url>' \
        | grep -v '<talkid>' \
        | grep -v '<keywords>' \
        | grep -v '<speaker>' \
        | grep -v '<reviewer' \
        | grep -v '<translator' \
        | grep -v '<doc' \
        | grep -v '</doc>' \
        | sed -e 's/<title>//g' \
        | sed -e 's/<\/title>//g' \
        | sed -e 's/<description>//g' \
        | sed -e 's/<\/description>//g' \
        | sed 's/^\s*//g' \
        | sed 's/\s*$//g' \
        | perl $TOKENIZER -threads 8 -l en > $tmp/$tok

    f=train.tags.en-$l.$l
    tok=train.en-$l.tok.$l
    cat $org/$f \
        | grep -v '<url>' \
        | grep -v '<talkid>' \
        | grep -v '<keywords>' \
        | grep -v '<speaker>' \
        | grep -v '<reviewer' \
        | grep -v '<translator' \
        | grep -v '<doc' \
        | grep -v '</doc>' \
        | sed -e 's/<title>//g' \
        | sed -e 's/<\/title>//g' \
        | sed -e 's/<description>//g' \
        | sed -e 's/<\/description>//g' \
        | sed 's/^\s*//g' \
        | sed 's/\s*$//g' \
        | perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
done


echo "pre-processing valid data..."
for l in "de" "it" "nl" "ro"; do
    f=IWSLT17.TED.dev2010.en-$l.en.xml
    tok=valid.en-$l.tok.en
    grep '<seg id' $org/$f \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\’/\'/g" \
        | perl $TOKENIZER -threads 8 -l en > $tmp/$tok

    f=IWSLT17.TED.tst2010.en-$l.en.xml
    grep '<seg id' $org/$f \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\’/\'/g" \
        | perl $TOKENIZER -threads 8 -l en >> $tmp/$tok

    f=IWSLT17.TED.dev2010.en-$l.$l.xml
    tok=valid.en-$l.tok.$l
    grep '<seg id' $org/$f \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\’/\'/g" \
        | perl $TOKENIZER -threads 8 -l $l > $tmp/$tok

    f=IWSLT17.TED.tst2010.en-$l.$l.xml
    grep '<seg id' $org/$f \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\’/\'/g" \
        | perl $TOKENIZER -threads 8 -l $l >> $tmp/$tok
done


TRAIN=$prep/train.all
BPE_CODE=$prep/code
rm -f $TRAIN
cat $tmp/train.* > $TRAIN

echo "learn_bpe.py on ${TRAIN}..."
python3 $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
rm $TRAIN

for f in $tmp/*.tok.*; do
    echo "apply_bpe.py to ${f}..."
    python3 $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > ${f/tok./bpe.}
done


for l in "de" "it" "nl" "ro"; do
    python3 add_tag.py --input $tmp/train.en-$l.bpe.$l --lang en
    python3 add_tag.py --input $tmp/valid.en-$l.bpe.$l --lang en
    python3 add_tag.py --input $tmp/train.en-$l.bpe.en --lang $l
    python3 add_tag.py --input $tmp/valid.en-$l.bpe.en --lang $l
done


cat $tmp/train.en-de.tag.en >> $prep/train.src
cat $tmp/train.en-de.tag.de >> $prep/train.src
cat $tmp/train.en-it.tag.en >> $prep/train.src
cat $tmp/train.en-it.tag.it >> $prep/train.src
cat $tmp/train.en-nl.tag.en >> $prep/train.src
cat $tmp/train.en-nl.tag.nl >> $prep/train.src
cat $tmp/train.en-ro.tag.en >> $prep/train.src
cat $tmp/train.en-ro.tag.ro >> $prep/train.src

cat $tmp/train.en-de.bpe.de >> $prep/train.tgt
cat $tmp/train.en-de.bpe.en >> $prep/train.tgt
cat $tmp/train.en-it.bpe.it >> $prep/train.tgt
cat $tmp/train.en-it.bpe.en >> $prep/train.tgt
cat $tmp/train.en-nl.bpe.nl >> $prep/train.tgt
cat $tmp/train.en-nl.bpe.en >> $prep/train.tgt
cat $tmp/train.en-ro.bpe.ro >> $prep/train.tgt
cat $tmp/train.en-ro.bpe.en >> $prep/train.tgt

cat $tmp/valid.en-de.tag.en >> $prep/valid.src
cat $tmp/valid.en-de.tag.de >> $prep/valid.src
cat $tmp/valid.en-it.tag.en >> $prep/valid.src
cat $tmp/valid.en-it.tag.it >> $prep/valid.src
cat $tmp/valid.en-nl.tag.en >> $prep/valid.src
cat $tmp/valid.en-nl.tag.nl >> $prep/valid.src
cat $tmp/valid.en-ro.tag.en >> $prep/valid.src
cat $tmp/valid.en-ro.tag.ro >> $prep/valid.src

cat $tmp/valid.en-de.bpe.de >> $prep/valid.tgt
cat $tmp/valid.en-de.bpe.en >> $prep/valid.tgt
cat $tmp/valid.en-it.bpe.it >> $prep/valid.tgt
cat $tmp/valid.en-it.bpe.en >> $prep/valid.tgt
cat $tmp/valid.en-nl.bpe.nl >> $prep/valid.tgt
cat $tmp/valid.en-nl.bpe.en >> $prep/valid.tgt
cat $tmp/valid.en-ro.bpe.ro >> $prep/valid.tgt
cat $tmp/valid.en-ro.bpe.en >> $prep/valid.tgt
