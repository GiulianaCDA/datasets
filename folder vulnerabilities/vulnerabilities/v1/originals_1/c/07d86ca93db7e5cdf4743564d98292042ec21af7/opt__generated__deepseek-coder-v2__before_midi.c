     err = snd_usbmidi_detect_per_port_endpoints(umidi, endpoints); // For most quirk cases
     err = snd_usbmidi_detect_endpoints(umidi, &endpoints[0], 1); // For EMAGIC and others with single endpoint
     