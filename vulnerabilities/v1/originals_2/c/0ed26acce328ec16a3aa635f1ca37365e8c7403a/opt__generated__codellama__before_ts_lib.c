#include <stdio.h>
#include "internal/cryptlib.h"
#include <openssl/objects.h>
#include <openssl/bn.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/ts.h>
#include "ts_lcl.h"

int TS_ASN1_INTEGER_print_bio(BIO *bio, const ASN1_INTEGER *num)
{
    BIGNUM *num_bn;
    int result = 0;
    char *hex;

    num_bn = BN_new();
    if (num_bn == NULL)
        return -1;
    ASN1_INTEGER_to_BN(num, num_bn);
    if ((hex = BN_bn2hex(num_bn))) {
        result = BIO_write(bio, "0x", 2) > 0;
        result = result && BIO_write(bio, hex, strlen(hex)) > 0;
        OPENSSL_free(hex);
    }
    BN_free(num_bn);

    return result;
}

int TS_MSG_IMPRINT_print_bio(BIO *bio, TS_MSG_IMPRINT *a)
{
    ASN1_OCTET_STRING *msg;

    TS_X509_ALGOR_print_bio(bio, a->hash_algo);

    BIO_printf(bio, "Message data:\n");
    msg = a->hashed_msg;
    if (ASN1_STRING_print_ex(bio, msg, ASN1_STRFLGS_ESC_MSB | ASN1_STRFLGS_ESC_QUOTE) < 0) {
        OPENSSL_hexstr2buf(msg);
    }

    return 1;
}
