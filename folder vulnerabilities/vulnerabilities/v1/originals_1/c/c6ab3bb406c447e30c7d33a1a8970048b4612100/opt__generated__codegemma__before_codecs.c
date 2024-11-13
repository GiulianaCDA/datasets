// Create regular expressions
regex_t re, re2;
// ... initialize regular expressions

// Find a stream with media type "audio"
int found = stream_find(msg, &re, NULL);

// Delete a stream with transport type "RTP/AVP"
int deleted = stream_delete(msg, &re, &re2);
