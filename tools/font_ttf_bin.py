# Read ttf and convert it to static const C array

TARGET: str = 'meow.ttf'
DST: str = 'meow.h'

def main():
    # read binary file and convert it to C array and pack 8 bytes into a 8-byte hex integer literal
    with open(TARGET, 'rb') as f:
        data = f.read()
        with open(DST, 'w') as dst:
            dst.write(f'static const uint8_t wl__font_data[{len(data)}] = {{')
            for i in range(len(data)):
                if i % 32 == 0:
                    dst.write('\n\t')
                dst.write(f'{data[i]},')
            dst.write('};')


if __name__ == '__main__':
    main()
