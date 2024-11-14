// Create a USB MIDI device with default settings
snd_usbmidi_create(usb_device, NULL);

// Create a USB MIDI device with custom quirks
snd_usbmidi_create(usb_device, &custom_quirks);
