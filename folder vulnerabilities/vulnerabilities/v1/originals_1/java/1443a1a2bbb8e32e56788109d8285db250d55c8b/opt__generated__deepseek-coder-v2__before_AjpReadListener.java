   bufferSize += rawMessages[i].length;
   final ByteBuffer buffer = ByteBuffer.allocate(bufferSize);
   for (int i = 0; i < rawMessages.length; i++) {
       buffer.put(rawMessages[i]);
   }
   buffer.flip();
   