# Read ttf and convert it to static const C array

TARGET: str = 'fonts/AnkaCoder-r.ttf'
DST: str = 'font.h'

def main():
    # read binary file and convert it to C array and pack 8 bytes into a 8-byte hex integer literal
    with open(TARGET, 'rb') as f:
        data = f.read()
        with open(DST, 'w') as dst:
            dst.write(f'const uint8_t mag__font_data[{len(data)}] = {{')
            for i in range(len(data)):
                if i % 32 == 0:
                    dst.write('\n\t')
                dst.write(f'{data[i]},')
            dst.write('};')


if __name__ == '__main__':
    main()
