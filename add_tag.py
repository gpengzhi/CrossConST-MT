import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--lang', required=True)
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    tag = '<2{}> '.format(args.lang)
    out = args.input.replace('bpe', 'tag')

    fout = open(out, 'w')
    for line in open(args.input, 'r').readlines():
        fout.write(tag + line)


if __name__ == '__main__':
    main()
