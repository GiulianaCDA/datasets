[PYTHON]
def snd_usbmidi_create(midi_list, umidi, usb_dev, iface, altsetting, quirk):
    # ...

    endpoint_count = iface->cur_altsetting->desc.bNumEndpoints
    endpoints = []
    for i in range(endpoint_count):
        endpoint = SndUSBMIDIEndpoint()
        endpoint.bEndpointAddress = usb_dev->actconfig->if_desc[iface->cur_altsetting - 1].endpoint + i
        endpoint.bmAttributes = usb_dev->actconfig->if_desc[iface->cur_altsetting - 1].endpoint[i].desc.bmAttributes
        endpoint.wMaxPacketSize = usb_dev->actconfig->if_desc[iface->cur_altsetting - 1].endpoint[i].desc.wMaxPacketSize
        endpoints.append(endpoint)

    # ...

    return 0
[/PYTHON]
