// Open an NNTP mailbox
struct Context *ctx = nntp_mbox_open(...);

// Find the children of article with Message-ID "msg-id"
int status = nntp_check_children(ctx, "msg-id");

// Close the NNTP mailbox
nntp_mbox_close(ctx);
