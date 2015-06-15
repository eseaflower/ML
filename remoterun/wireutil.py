import struct
import socket

def pack_string(data):
    """
        data: String to be packed for transfer
    """  
    return struct.pack("I", len(data)) + data.encode()
    
def write_retry(s, data):
    total_sent = 0
    to_send = len(data)
    while total_sent < to_send:
        sent = s.send(data[total_sent:])
        if sent <= 0:
            raise EOFError("Network error")
        total_sent += sent


def read_retry(s, size):
    result = bytes()
    while size > 0:
        segment = s.recv(size)
        if not segment:
            raise EOFError("End of data")
        result += segment
        size -= len(segment)
    return result

def read_int(s):
    int_buffer = read_retry(s, 4)
    return struct.unpack("I", int_buffer)[0]

def write_packed_string(s, data):
    packed_data = pack_string(data)
    write_retry(s, packed_data)

def read_packed_string(s):
    # Read packed size
    message_length = read_int(s)
    rawString = read_retry(s, message_length)    
    return rawString.decode()

def read_binary_item(s):
    id = read_packed_string(s)
    length = read_int(s)
    data = read_retry(s, length)
    return id, data

def read_binary_data(s):
    context = None
    item_count = read_int(s)
    if item_count > 0:
        context = dict()
        for i in range(item_count):
            id, data = read_binary_item(s)
            context[id] = data
    return context