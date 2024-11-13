 
[PYTHON]
def mux(input, output, dst):
    with open(output, "wb") as fout:
        with open(input, "rb") as fin:
            while True:
                data = fin.read(188)
                if not data:
                    break
                fout.write(data[:3])
                payload = data[4:]
                if payload[0] & 0x40 == 0 or (payload[0] & 0x20 and payload[5] != 1):
                    dst.write(payload)
                    continue
                pes_header_length = payload[1] & 7
                pes_header_data = payload[2:2 + pes_header_length]
                if len(pes_header_data) < 5 or pes_header_data[0] != 0 or (pes_header_data[3] & 0xC8) != 0x80:
                    dst.write(payload)
                    continue
                pts = pes_header_data[4] & 1 == 1
                dts = pts and (pes_header_data[7] >> 6 & 1) or 0
                if not (pts or dts):
                    dst.write(payload)
                    continue
                ts = int.from_bytes(payload[:3], 'big')
                scramble = payload[3] >> 6
                adaptation = payload[3] & 3
                payload = payload[4 + pes_header_length:]
                if pts:
                    pts_value = int.from_bytes(payload[:5], 'big')
                    dst.write(bytearray([0x50, 0x10, 0x00, ts >> 24 & 0xFF, (ts >> 16) & 0xFF, (ts >> 8) & 0xFF, ts & 0xFF,
                                        0x80, 0x07, 0x80, pts_value >> 29 & 0x7F, pts_value >> 14 & 0x7FFF, pts_value << 3 & 0xFFFFFFFF]))
                    payload = payload[5:]
                if dts:
                    dts_value = int.from_bytes(payload[:5], 'big')
                    dst.write(bytearray([0x50, 0x11, 0x00, ts >> 24 & 0xFF, (ts >> 16) & 0xFF, (ts >> 8) & 0xFF, ts & 0xFF,
                                        0x80, 0x07, 0x80, dts_value >> 29 & 0x7F, dts_value >> 14 & 0x7FFF, dts_value << 3 & 0xFFFFFFFF]))
                    payload = payload[5:]
                dst.write(bytearray([0x50, adaptation | scramble << 6, len(payload) + 4]))
                dst.write(payload)
[/PYTHON]
