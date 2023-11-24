/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*!
 * Provides the implementation for all GH100 SPDM certificate HAL interfaces.
 */

/* ------------------------ Includes --------------------------------------- */
#include "nvRmReg.h"
#include "gpu/spdm/spdm.h"
#include "gpu/spdm/libspdm_includes.h"
#include "spdm/rmspdmvendordef.h"
#include "flcnretval.h"

/* ------------------------ Macros ----------------------------------------- */
#define DER_LONG_FORM_LENGTH_FIELD_BIT   (0x80)
#define DER_CERT_SIZE_FIELD_LENGTH       (0x4)

#define SPDM_MAX_ENCODED_CERT_CHAIN_SIZE (0x1400)

#define SPDM_PEM_BEGIN_CERTIFICATE "-----BEGIN CERTIFICATE-----\n"
#define SPDM_PEM_END_CERTIFICATE   "-----END CERTIFICATE-----\n"

#define SPDM_L1_CERTIFICATE_PEM "-----BEGIN CERTIFICATE-----\n"\
                                "MIICCzCCAZCgAwIBAgIQLTZwscoQBBHB/sDoKgZbVDAKBggqhkjOPQQDAzA1MSIw\n"\
                                "IAYDVQQDDBlOVklESUEgRGV2aWNlIElkZW50aXR5IENBMQ8wDQYDVQQKDAZOVklE\n"\
                                "SUEwIBcNMjExMTA1MDAwMDAwWhgPOTk5OTEyMzEyMzU5NTlaMDUxIjAgBgNVBAMM\n"\
                                "GU5WSURJQSBEZXZpY2UgSWRlbnRpdHkgQ0ExDzANBgNVBAoMBk5WSURJQTB2MBAG\n"\
                                "ByqGSM49AgEGBSuBBAAiA2IABA5MFKM7+KViZljbQSlgfky/RRnEQScW9NDZF8SX\n"\
                                "gAW96r6u/Ve8ZggtcYpPi2BS4VFu6KfEIrhN6FcHG7WP05W+oM+hxj7nyA1r1jkB\n"\
                                "2Ry70YfThX3Ba1zOryOP+MJ9vaNjMGEwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8B\n"\
                                "Af8EBAMCAQYwHQYDVR0OBBYEFFeF/4PyY8xlfWi3Olv0jUrL+0lfMB8GA1UdIwQY\n"\
                                "MBaAFFeF/4PyY8xlfWi3Olv0jUrL+0lfMAoGCCqGSM49BAMDA2kAMGYCMQCPeFM3\n"\
                                "TASsKQVaT+8S0sO9u97PVGCpE9d/I42IT7k3UUOLSR/qvJynVOD1vQKVXf0CMQC+\n"\
                                "EY55WYoDBvs2wPAH1Gw4LbcwUN8QCff8bFmV4ZxjCRr4WXTLFHBKjbfneGSBWwA=\n"\
                                "-----END CERTIFICATE-----\n"

#define SPDM_L2_CERTIFICATE_PEM "-----BEGIN CERTIFICATE-----\n"\
                                "MIICijCCAhCgAwIBAgIQTCVe3jvQAb8/SjtgX8qJijAKBggqhkjOPQQDAzA1MSIw\n"\
                                "IAYDVQQDDBlOVklESUEgRGV2aWNlIElkZW50aXR5IENBMQ8wDQYDVQQKDAZOVklE\n"\
                                "SUEwIBcNMjIwMTEyMDAwMDAwWhgPOTk5OTEyMzEyMzU5NTlaMD0xHjAcBgNVBAMM\n"\
                                "FU5WSURJQSBHSDEwMCBJZGVudGl0eTEbMBkGA1UECgwSTlZJRElBIENvcnBvcmF0\n"\
                                "aW9uMHYwEAYHKoZIzj0CAQYFK4EEACIDYgAE+pg+tDUuILlZILk5wg22YEJ9Oh6c\n"\
                                "yPcsv3IvgRWcV4LeZK1pTCoQDIplZ0E4qsLG3G04pxsbMhxbqkiz9pqlTV2rtuVg\n"\
                                "SmIqnSYkU1jWXsPS9oVLCGE8VRLl1JvqyOxUo4HaMIHXMA8GA1UdEwEB/wQFMAMB\n"\
                                "Af8wDgYDVR0PAQH/BAQDAgEGMDsGA1UdHwQ0MDIwMKAuoCyGKmh0dHA6Ly9jcmwu\n"\
                                "bmRpcy5udmlkaWEuY29tL2NybC9sMS1yb290LmNybDA3BggrBgEFBQcBAQQrMCkw\n"\
                                "JwYIKwYBBQUHMAGGG2h0dHA6Ly9vY3NwLm5kaXMubnZpZGlhLmNvbTAdBgNVHQ4E\n"\
                                "FgQUB0Kg6wOcgGB7oUFhmU2uJffCmx4wHwYDVR0jBBgwFoAUV4X/g/JjzGV9aLc6\n"\
                                "W/SNSsv7SV8wCgYIKoZIzj0EAwMDaAAwZQIxAPIQhnveFxYIrPzBqViT2I34SfS4\n"\
                                "JGWFnk/1UcdmgJmp+7l6rH/C4qxwntYSgeYrlQIwdjQuofHnhd1RL09OBO34566J\n"\
                                "C9bYAosT/86cCojiGjhLnal9hJOH0nS/lrbaoc5a\n"\
                                "-----END CERTIFICATE-----\n"

#define SPDM_L3_CERTIFICATE_PEM "-----BEGIN CERTIFICATE-----\n"\
                                "MIICqjCCAi+gAwIBAgIQav5xhPkiMsjfeyQiYXduVjAKBggqhkjOPQQDAzA9MR4w\n"\
                                "HAYDVQQDDBVOVklESUEgR0gxMDAgSWRlbnRpdHkxGzAZBgNVBAoMEk5WSURJQSBD\n"\
                                "b3Jwb3JhdGlvbjAgFw0yMjAzMDEwMDAwMDBaGA85OTk5MTIzMTIzNTk1OVowUzEn\n"\
                                "MCUGA1UEAwweTlZJRElBIEdIMTAwIFByb3Zpc2lvbmVyIElDQSAxMRswGQYDVQQK\n"\
                                "DBJOVklESUEgQ29ycG9yYXRpb24xCzAJBgNVBAYTAlVTMHYwEAYHKoZIzj0CAQYF\n"\
                                "K4EEACIDYgAEzUdWqjn1OlXhLfFOKAFTghqG+Q3zF4xgSBbZsUEyWYCC3rKjE9Nn\n"\
                                "o88ZpBQx85Oo0PkqP2dwoMVNTQMv5cvy9jLaTvSTXZwN2HQHE9u7x7BIYrWi0sG3\n"\
                                "5q1IJNSOGO5Lo4HbMIHYMA8GA1UdEwEB/wQFMAMBAf8wDgYDVR0PAQH/BAQDAgEG\n"\
                                "MDwGA1UdHwQ1MDMwMaAvoC2GK2h0dHA6Ly9jcmwubmRpcy5udmlkaWEuY29tL2Ny\n"\
                                "bC9sMi1naDEwMC5jcmwwNwYIKwYBBQUHAQEEKzApMCcGCCsGAQUFBzABhhtodHRw\n"\
                                "Oi8vb2NzcC5uZGlzLm52aWRpYS5jb20wHQYDVR0OBBYEFCloyxYs0HeVcqJ5EAPm\n"\
                                "nroMzAqUMB8GA1UdIwQYMBaAFAdCoOsDnIBge6FBYZlNriX3wpseMAoGCCqGSM49\n"\
                                "BAMDA2kAMGYCMQDK0BCr49DNJ48Yh5wu388bZifDFxAsiUS4U1fGmpJZFhCbODH6\n"\
                                "mRwcMxp6EOayZuYCMQDYKTyNc2FxWFuhHtdCE3ls4S7SInehdErTZNuhFymc4YOM\n"\
                                "6VlLWTY/CM+resjjqxQ=\n"\
                                "-----END CERTIFICATE-----\n"

const static NvU8 SPDM_L1_CERTIFICATE_DER[527] =
{
    0x30, 0x82, 0x02, 0x0b, 0x30, 0x82, 0x01, 0x90, 0xa0, 0x03, 0x02, 0x01, 0x02, 0x02, 0x10, 0x2d,
    0x36, 0x70, 0xb1, 0xca, 0x10, 0x04, 0x11, 0xc1, 0xfe, 0xc0, 0xe8, 0x2a, 0x06, 0x5b, 0x54, 0x30,
    0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x03, 0x30, 0x35, 0x31, 0x22, 0x30,
    0x20, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c, 0x19, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20, 0x44,
    0x65, 0x76, 0x69, 0x63, 0x65, 0x20, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x20, 0x43,
    0x41, 0x31, 0x0f, 0x30, 0x0d, 0x06, 0x03, 0x55, 0x04, 0x0a, 0x0c, 0x06, 0x4e, 0x56, 0x49, 0x44,
    0x49, 0x41, 0x30, 0x20, 0x17, 0x0d, 0x32, 0x31, 0x31, 0x31, 0x30, 0x35, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x5a, 0x18, 0x0f, 0x39, 0x39, 0x39, 0x39, 0x31, 0x32, 0x33, 0x31, 0x32, 0x33, 0x35,
    0x39, 0x35, 0x39, 0x5a, 0x30, 0x35, 0x31, 0x22, 0x30, 0x20, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c,
    0x19, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x20, 0x49,
    0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x20, 0x43, 0x41, 0x31, 0x0f, 0x30, 0x0d, 0x06, 0x03,
    0x55, 0x04, 0x0a, 0x0c, 0x06, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x30, 0x76, 0x30, 0x10, 0x06,
    0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x05, 0x2b, 0x81, 0x04, 0x00, 0x22, 0x03,
    0x62, 0x00, 0x04, 0x0e, 0x4c, 0x14, 0xa3, 0x3b, 0xf8, 0xa5, 0x62, 0x66, 0x58, 0xdb, 0x41, 0x29,
    0x60, 0x7e, 0x4c, 0xbf, 0x45, 0x19, 0xc4, 0x41, 0x27, 0x16, 0xf4, 0xd0, 0xd9, 0x17, 0xc4, 0x97,
    0x80, 0x05, 0xbd, 0xea, 0xbe, 0xae, 0xfd, 0x57, 0xbc, 0x66, 0x08, 0x2d, 0x71, 0x8a, 0x4f, 0x8b,
    0x60, 0x52, 0xe1, 0x51, 0x6e, 0xe8, 0xa7, 0xc4, 0x22, 0xb8, 0x4d, 0xe8, 0x57, 0x07, 0x1b, 0xb5,
    0x8f, 0xd3, 0x95, 0xbe, 0xa0, 0xcf, 0xa1, 0xc6, 0x3e, 0xe7, 0xc8, 0x0d, 0x6b, 0xd6, 0x39, 0x01,
    0xd9, 0x1c, 0xbb, 0xd1, 0x87, 0xd3, 0x85, 0x7d, 0xc1, 0x6b, 0x5c, 0xce, 0xaf, 0x23, 0x8f, 0xf8,
    0xc2, 0x7d, 0xbd, 0xa3, 0x63, 0x30, 0x61, 0x30, 0x0f, 0x06, 0x03, 0x55, 0x1d, 0x13, 0x01, 0x01,
    0xff, 0x04, 0x05, 0x30, 0x03, 0x01, 0x01, 0xff, 0x30, 0x0e, 0x06, 0x03, 0x55, 0x1d, 0x0f, 0x01,
    0x01, 0xff, 0x04, 0x04, 0x03, 0x02, 0x01, 0x06, 0x30, 0x1d, 0x06, 0x03, 0x55, 0x1d, 0x0e, 0x04,
    0x16, 0x04, 0x14, 0x57, 0x85, 0xff, 0x83, 0xf2, 0x63, 0xcc, 0x65, 0x7d, 0x68, 0xb7, 0x3a, 0x5b,
    0xf4, 0x8d, 0x4a, 0xcb, 0xfb, 0x49, 0x5f, 0x30, 0x1f, 0x06, 0x03, 0x55, 0x1d, 0x23, 0x04, 0x18,
    0x30, 0x16, 0x80, 0x14, 0x57, 0x85, 0xff, 0x83, 0xf2, 0x63, 0xcc, 0x65, 0x7d, 0x68, 0xb7, 0x3a,
    0x5b, 0xf4, 0x8d, 0x4a, 0xcb, 0xfb, 0x49, 0x5f, 0x30, 0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce,
    0x3d, 0x04, 0x03, 0x03, 0x03, 0x69, 0x00, 0x30, 0x66, 0x02, 0x31, 0x00, 0x8f, 0x78, 0x53, 0x37,
    0x4c, 0x04, 0xac, 0x29, 0x05, 0x5a, 0x4f, 0xef, 0x12, 0xd2, 0xc3, 0xbd, 0xbb, 0xde, 0xcf, 0x54,
    0x60, 0xa9, 0x13, 0xd7, 0x7f, 0x23, 0x8d, 0x88, 0x4f, 0xb9, 0x37, 0x51, 0x43, 0x8b, 0x49, 0x1f,
    0xea, 0xbc, 0x9c, 0xa7, 0x54, 0xe0, 0xf5, 0xbd, 0x02, 0x95, 0x5d, 0xfd, 0x02, 0x31, 0x00, 0xbe,
    0x11, 0x8e, 0x79, 0x59, 0x8a, 0x03, 0x06, 0xfb, 0x36, 0xc0, 0xf0, 0x07, 0xd4, 0x6c, 0x38, 0x2d,
    0xb7, 0x30, 0x50, 0xdf, 0x10, 0x09, 0xf7, 0xfc, 0x6c, 0x59, 0x95, 0xe1, 0x9c, 0x63, 0x09, 0x1a,
    0xf8, 0x59, 0x74, 0xcb, 0x14, 0x70, 0x4a, 0x8d, 0xb7, 0xe7, 0x78, 0x64, 0x81, 0x5b, 0x00
};

const static NvU8 SPDM_L2_CERTIFICATE_DER[654] =
{
    0x30, 0x82, 0x02, 0x8a, 0x30, 0x82, 0x02, 0x10, 0xa0, 0x03, 0x02, 0x01, 0x02, 0x02, 0x10, 0x4c,
    0x25, 0x5e, 0xde, 0x3b, 0xd0, 0x01, 0xbf, 0x3f, 0x4a, 0x3b, 0x60, 0x5f, 0xca, 0x89, 0x8a, 0x30,
    0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x03, 0x30, 0x35, 0x31, 0x22, 0x30,
    0x20, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c, 0x19, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20, 0x44,
    0x65, 0x76, 0x69, 0x63, 0x65, 0x20, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x20, 0x43,
    0x41, 0x31, 0x0f, 0x30, 0x0d, 0x06, 0x03, 0x55, 0x04, 0x0a, 0x0c, 0x06, 0x4e, 0x56, 0x49, 0x44,
    0x49, 0x41, 0x30, 0x20, 0x17, 0x0d, 0x32, 0x32, 0x30, 0x31, 0x31, 0x32, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x30, 0x5a, 0x18, 0x0f, 0x39, 0x39, 0x39, 0x39, 0x31, 0x32, 0x33, 0x31, 0x32, 0x33, 0x35,
    0x39, 0x35, 0x39, 0x5a, 0x30, 0x3d, 0x31, 0x1e, 0x30, 0x1c, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c,
    0x15, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20, 0x47, 0x48, 0x31, 0x30, 0x30, 0x20, 0x49, 0x64,
    0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x31, 0x1b, 0x30, 0x19, 0x06, 0x03, 0x55, 0x04, 0x0a, 0x0c,
    0x12, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20, 0x43, 0x6f, 0x72, 0x70, 0x6f, 0x72, 0x61, 0x74,
    0x69, 0x6f, 0x6e, 0x30, 0x76, 0x30, 0x10, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01,
    0x06, 0x05, 0x2b, 0x81, 0x04, 0x00, 0x22, 0x03, 0x62, 0x00, 0x04, 0xfa, 0x98, 0x3e, 0xb4, 0x35,
    0x2e, 0x20, 0xb9, 0x59, 0x20, 0xb9, 0x39, 0xc2, 0x0d, 0xb6, 0x60, 0x42, 0x7d, 0x3a, 0x1e, 0x9c,
    0xc8, 0xf7, 0x2c, 0xbf, 0x72, 0x2f, 0x81, 0x15, 0x9c, 0x57, 0x82, 0xde, 0x64, 0xad, 0x69, 0x4c,
    0x2a, 0x10, 0x0c, 0x8a, 0x65, 0x67, 0x41, 0x38, 0xaa, 0xc2, 0xc6, 0xdc, 0x6d, 0x38, 0xa7, 0x1b,
    0x1b, 0x32, 0x1c, 0x5b, 0xaa, 0x48, 0xb3, 0xf6, 0x9a, 0xa5, 0x4d, 0x5d, 0xab, 0xb6, 0xe5, 0x60,
    0x4a, 0x62, 0x2a, 0x9d, 0x26, 0x24, 0x53, 0x58, 0xd6, 0x5e, 0xc3, 0xd2, 0xf6, 0x85, 0x4b, 0x08,
    0x61, 0x3c, 0x55, 0x12, 0xe5, 0xd4, 0x9b, 0xea, 0xc8, 0xec, 0x54, 0xa3, 0x81, 0xda, 0x30, 0x81,
    0xd7, 0x30, 0x0f, 0x06, 0x03, 0x55, 0x1d, 0x13, 0x01, 0x01, 0xff, 0x04, 0x05, 0x30, 0x03, 0x01,
    0x01, 0xff, 0x30, 0x0e, 0x06, 0x03, 0x55, 0x1d, 0x0f, 0x01, 0x01, 0xff, 0x04, 0x04, 0x03, 0x02,
    0x01, 0x06, 0x30, 0x3b, 0x06, 0x03, 0x55, 0x1d, 0x1f, 0x04, 0x34, 0x30, 0x32, 0x30, 0x30, 0xa0,
    0x2e, 0xa0, 0x2c, 0x86, 0x2a, 0x68, 0x74, 0x74, 0x70, 0x3a, 0x2f, 0x2f, 0x63, 0x72, 0x6c, 0x2e,
    0x6e, 0x64, 0x69, 0x73, 0x2e, 0x6e, 0x76, 0x69, 0x64, 0x69, 0x61, 0x2e, 0x63, 0x6f, 0x6d, 0x2f,
    0x63, 0x72, 0x6c, 0x2f, 0x6c, 0x31, 0x2d, 0x72, 0x6f, 0x6f, 0x74, 0x2e, 0x63, 0x72, 0x6c, 0x30,
    0x37, 0x06, 0x08, 0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x01, 0x01, 0x04, 0x2b, 0x30, 0x29, 0x30,
    0x27, 0x06, 0x08, 0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x30, 0x01, 0x86, 0x1b, 0x68, 0x74, 0x74,
    0x70, 0x3a, 0x2f, 0x2f, 0x6f, 0x63, 0x73, 0x70, 0x2e, 0x6e, 0x64, 0x69, 0x73, 0x2e, 0x6e, 0x76,
    0x69, 0x64, 0x69, 0x61, 0x2e, 0x63, 0x6f, 0x6d, 0x30, 0x1d, 0x06, 0x03, 0x55, 0x1d, 0x0e, 0x04,
    0x16, 0x04, 0x14, 0x07, 0x42, 0xa0, 0xeb, 0x03, 0x9c, 0x80, 0x60, 0x7b, 0xa1, 0x41, 0x61, 0x99,
    0x4d, 0xae, 0x25, 0xf7, 0xc2, 0x9b, 0x1e, 0x30, 0x1f, 0x06, 0x03, 0x55, 0x1d, 0x23, 0x04, 0x18,
    0x30, 0x16, 0x80, 0x14, 0x57, 0x85, 0xff, 0x83, 0xf2, 0x63, 0xcc, 0x65, 0x7d, 0x68, 0xb7, 0x3a,
    0x5b, 0xf4, 0x8d, 0x4a, 0xcb, 0xfb, 0x49, 0x5f, 0x30, 0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce,
    0x3d, 0x04, 0x03, 0x03, 0x03, 0x68, 0x00, 0x30, 0x65, 0x02, 0x31, 0x00, 0xf2, 0x10, 0x86, 0x7b,
    0xde, 0x17, 0x16, 0x08, 0xac, 0xfc, 0xc1, 0xa9, 0x58, 0x93, 0xd8, 0x8d, 0xf8, 0x49, 0xf4, 0xb8,
    0x24, 0x65, 0x85, 0x9e, 0x4f, 0xf5, 0x51, 0xc7, 0x66, 0x80, 0x99, 0xa9, 0xfb, 0xb9, 0x7a, 0xac,
    0x7f, 0xc2, 0xe2, 0xac, 0x70, 0x9e, 0xd6, 0x12, 0x81, 0xe6, 0x2b, 0x95, 0x02, 0x30, 0x76, 0x34,
    0x2e, 0xa1, 0xf1, 0xe7, 0x85, 0xdd, 0x51, 0x2f, 0x4f, 0x4e, 0x04, 0xed, 0xf8, 0xe7, 0xae, 0x89,
    0x0b, 0xd6, 0xd8, 0x02, 0x8b, 0x13, 0xff, 0xce, 0x9c, 0x0a, 0x88, 0xe2, 0x1a, 0x38, 0x4b, 0x9d,
    0xa9, 0x7d, 0x84, 0x93, 0x87, 0xd2, 0x74, 0xbf, 0x96, 0xb6, 0xda, 0xa1, 0xce, 0x5a,
};

const static NvU8 SPDM_L3_CERTIFICATE_DER[686] =
{
    0x30, 0x82, 0x02, 0xaa, 0x30, 0x82, 0x02, 0x2f, 0xa0, 0x03, 0x02, 0x01, 0x02, 0x02, 0x10, 0x6a,
    0xfe, 0x71, 0x84, 0xf9, 0x22, 0x32, 0xc8, 0xdf, 0x7b, 0x24, 0x22, 0x61, 0x77, 0x6e, 0x56, 0x30,
    0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x03, 0x30, 0x3d, 0x31, 0x1e, 0x30,
    0x1c, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c, 0x15, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20, 0x47,
    0x48, 0x31, 0x30, 0x30, 0x20, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x31, 0x1b, 0x30,
    0x19, 0x06, 0x03, 0x55, 0x04, 0x0a, 0x0c, 0x12, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20, 0x43,
    0x6f, 0x72, 0x70, 0x6f, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x30, 0x20, 0x17, 0x0d, 0x32, 0x32,
    0x30, 0x33, 0x30, 0x31, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x5a, 0x18, 0x0f, 0x39, 0x39, 0x39,
    0x39, 0x31, 0x32, 0x33, 0x31, 0x32, 0x33, 0x35, 0x39, 0x35, 0x39, 0x5a, 0x30, 0x53, 0x31, 0x27,
    0x30, 0x25, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c, 0x1e, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20,
    0x47, 0x48, 0x31, 0x30, 0x30, 0x20, 0x50, 0x72, 0x6f, 0x76, 0x69, 0x73, 0x69, 0x6f, 0x6e, 0x65,
    0x72, 0x20, 0x49, 0x43, 0x41, 0x20, 0x31, 0x31, 0x1b, 0x30, 0x19, 0x06, 0x03, 0x55, 0x04, 0x0a,
    0x0c, 0x12, 0x4e, 0x56, 0x49, 0x44, 0x49, 0x41, 0x20, 0x43, 0x6f, 0x72, 0x70, 0x6f, 0x72, 0x61,
    0x74, 0x69, 0x6f, 0x6e, 0x31, 0x0b, 0x30, 0x09, 0x06, 0x03, 0x55, 0x04, 0x06, 0x13, 0x02, 0x55,
    0x53, 0x30, 0x76, 0x30, 0x10, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x05,
    0x2b, 0x81, 0x04, 0x00, 0x22, 0x03, 0x62, 0x00, 0x04, 0xcd, 0x47, 0x56, 0xaa, 0x39, 0xf5, 0x3a,
    0x55, 0xe1, 0x2d, 0xf1, 0x4e, 0x28, 0x01, 0x53, 0x82, 0x1a, 0x86, 0xf9, 0x0d, 0xf3, 0x17, 0x8c,
    0x60, 0x48, 0x16, 0xd9, 0xb1, 0x41, 0x32, 0x59, 0x80, 0x82, 0xde, 0xb2, 0xa3, 0x13, 0xd3, 0x67,
    0xa3, 0xcf, 0x19, 0xa4, 0x14, 0x31, 0xf3, 0x93, 0xa8, 0xd0, 0xf9, 0x2a, 0x3f, 0x67, 0x70, 0xa0,
    0xc5, 0x4d, 0x4d, 0x03, 0x2f, 0xe5, 0xcb, 0xf2, 0xf6, 0x32, 0xda, 0x4e, 0xf4, 0x93, 0x5d, 0x9c,
    0x0d, 0xd8, 0x74, 0x07, 0x13, 0xdb, 0xbb, 0xc7, 0xb0, 0x48, 0x62, 0xb5, 0xa2, 0xd2, 0xc1, 0xb7,
    0xe6, 0xad, 0x48, 0x24, 0xd4, 0x8e, 0x18, 0xee, 0x4b, 0xa3, 0x81, 0xdb, 0x30, 0x81, 0xd8, 0x30,
    0x0f, 0x06, 0x03, 0x55, 0x1d, 0x13, 0x01, 0x01, 0xff, 0x04, 0x05, 0x30, 0x03, 0x01, 0x01, 0xff,
    0x30, 0x0e, 0x06, 0x03, 0x55, 0x1d, 0x0f, 0x01, 0x01, 0xff, 0x04, 0x04, 0x03, 0x02, 0x01, 0x06,
    0x30, 0x3c, 0x06, 0x03, 0x55, 0x1d, 0x1f, 0x04, 0x35, 0x30, 0x33, 0x30, 0x31, 0xa0, 0x2f, 0xa0,
    0x2d, 0x86, 0x2b, 0x68, 0x74, 0x74, 0x70, 0x3a, 0x2f, 0x2f, 0x63, 0x72, 0x6c, 0x2e, 0x6e, 0x64,
    0x69, 0x73, 0x2e, 0x6e, 0x76, 0x69, 0x64, 0x69, 0x61, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x63, 0x72,
    0x6c, 0x2f, 0x6c, 0x32, 0x2d, 0x67, 0x68, 0x31, 0x30, 0x30, 0x2e, 0x63, 0x72, 0x6c, 0x30, 0x37,
    0x06, 0x08, 0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x01, 0x01, 0x04, 0x2b, 0x30, 0x29, 0x30, 0x27,
    0x06, 0x08, 0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x30, 0x01, 0x86, 0x1b, 0x68, 0x74, 0x74, 0x70,
    0x3a, 0x2f, 0x2f, 0x6f, 0x63, 0x73, 0x70, 0x2e, 0x6e, 0x64, 0x69, 0x73, 0x2e, 0x6e, 0x76, 0x69,
    0x64, 0x69, 0x61, 0x2e, 0x63, 0x6f, 0x6d, 0x30, 0x1d, 0x06, 0x03, 0x55, 0x1d, 0x0e, 0x04, 0x16,
    0x04, 0x14, 0x29, 0x68, 0xcb, 0x16, 0x2c, 0xd0, 0x77, 0x95, 0x72, 0xa2, 0x79, 0x10, 0x03, 0xe6,
    0x9e, 0xba, 0x0c, 0xcc, 0x0a, 0x94, 0x30, 0x1f, 0x06, 0x03, 0x55, 0x1d, 0x23, 0x04, 0x18, 0x30,
    0x16, 0x80, 0x14, 0x07, 0x42, 0xa0, 0xeb, 0x03, 0x9c, 0x80, 0x60, 0x7b, 0xa1, 0x41, 0x61, 0x99,
    0x4d, 0xae, 0x25, 0xf7, 0xc2, 0x9b, 0x1e, 0x30, 0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d,
    0x04, 0x03, 0x03, 0x03, 0x69, 0x00, 0x30, 0x66, 0x02, 0x31, 0x00, 0xca, 0xd0, 0x10, 0xab, 0xe3,
    0xd0, 0xcd, 0x27, 0x8f, 0x18, 0x87, 0x9c, 0x2e, 0xdf, 0xcf, 0x1b, 0x66, 0x27, 0xc3, 0x17, 0x10,
    0x2c, 0x89, 0x44, 0xb8, 0x53, 0x57, 0xc6, 0x9a, 0x92, 0x59, 0x16, 0x10, 0x9b, 0x38, 0x31, 0xfa,
    0x99, 0x1c, 0x1c, 0x33, 0x1a, 0x7a, 0x10, 0xe6, 0xb2, 0x66, 0xe6, 0x02, 0x31, 0x00, 0xd8, 0x29,
    0x3c, 0x8d, 0x73, 0x61, 0x71, 0x58, 0x5b, 0xa1, 0x1e, 0xd7, 0x42, 0x13, 0x79, 0x6c, 0xe1, 0x2e,
    0xd2, 0x22, 0x77, 0xa1, 0x74, 0x4a, 0xd3, 0x64, 0xdb, 0xa1, 0x17, 0x29, 0x9c, 0xe1, 0x83, 0x8c,
    0xe9, 0x59, 0x4b, 0x59, 0x36, 0x3f, 0x08, 0xcf, 0xab, 0x7a, 0xc8, 0xe3, 0xab, 0x14,
};

/* ------------------------ Static Functions ------------------------------- */
/*!
 @param pCert       : The pointer to certification chain start
 @param bufferEnd   : The pointer to certification chain end
 @parsm pCertLength : The pointer to store return certification size

 @return Return NV-OK if no error.

* Static function that calculates the length of the X509 certificate in DER/TLV
* format. It assumes that the certificate is valid.
*/
static NV_STATUS
_calcX509CertSize
(
    NvU8 *pCert,
    NvU8 *bufferEnd,
    NvU32 *pCertLength
)
{
    // The cert is in TLV format.
    NvU32 certSize       = pCert[1];

    // Check to make sure that some data exists after SPDM header, and it is enough to check cert size.
    if (pCert + DER_CERT_SIZE_FIELD_LENGTH >= bufferEnd ||
        pCert + DER_CERT_SIZE_FIELD_LENGTH <= pCert)
    {
        NV_PRINTF(LEVEL_ERROR, " %s: pCert + DER_CERT_SIZE_FIELD_LENGTH(0x%x) is not valid value !! \n",
                  __FUNCTION__, DER_CERT_SIZE_FIELD_LENGTH);

       return NV_ERR_BUFFER_TOO_SMALL;
    }

    // Check if the length is in DER longform.
    // MSB in the length field is set for long form notation.
    // fields.
    if (certSize & DER_LONG_FORM_LENGTH_FIELD_BIT)
    {
        //
        // The remaining bits in the length field indicate the
        // number of following bytes used to represent the length.
        // in base 256, most significant digit first.
        //
        NvU32 numLenBytes = certSize & 0x3f;
        NvU8 *pStart      = &pCert[2];
        NvU8 *pEnd        = pStart + numLenBytes; // NOTE: Don't need to subtract numLenBytes 1 here.

        // Checking for buffer overflow.
        if (pEnd > bufferEnd)
        {
            return NV_ERR_BUFFER_TOO_SMALL;
        }

        certSize = *pStart;
        while (++pStart < pEnd)
        {
            certSize = (certSize << 8) + *pStart ;
        }
        // Total cert length includes the Tag + length
        // Adding it here.
        certSize += 2 + numLenBytes;
    }

    //
    // Check to make sure we have not hit end of buffer, and there is space for AK cert.
    // Check for underflow as well. This makes sure we haven't missed the calculation to
    // go past the end of the buffer
    //
    if (pCert + (certSize - 1) > bufferEnd ||
        pCert + (certSize - 1) <= pCert)
    {
        NV_PRINTF(LEVEL_ERROR, " %s: pCert + (certSize(0x%x) - 1) is not a valid value !! \n",
                  __FUNCTION__, certSize);

        return NV_ERR_BUFFER_TOO_SMALL;
    }

    *pCertLength = certSize;
    return NV_OK;
}

static NV_STATUS
pem_write_buffer
(
    NvU8 const *der,
    NvU64       derLen,
    NvU8       *buffer,
    NvU64       bufferLen,
    NvU64      *bufferUsed
)
{
    static const NvU8 base64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    NvU64 i, tmp, size;
    NvU64 printed = 0;
    NvU8 *ptr = buffer;

    // Base64 encoded size
    size = (derLen + 2) / 3 * 4;

    // Add 1 byte per 64 for newline
    size = size + (size + 63) / 64;

    // Add header excluding the terminating null and footer including the null
    size += sizeof(SPDM_PEM_BEGIN_CERTIFICATE) - 1 +
            sizeof(SPDM_PEM_END_CERTIFICATE);

    if (bufferLen < size)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    portMemCopy(ptr, bufferLen - (ptr - buffer), SPDM_PEM_BEGIN_CERTIFICATE,
                sizeof(SPDM_PEM_BEGIN_CERTIFICATE) - 1);
    ptr += sizeof(SPDM_PEM_BEGIN_CERTIFICATE) - 1;

    for (i = 0; (i + 2) < derLen; i += 3)
    {
        tmp = (der[i] << 16) | (der[i + 1] << 8) | (der[i + 2]);
        *ptr++ = base64[(tmp >> 18) & 63];
        *ptr++ = base64[(tmp >> 12) & 63];
        *ptr++ = base64[(tmp >> 6) & 63];
        *ptr++ = base64[(tmp >> 0) & 63];

        printed += 4;
        if (printed == 64)
        {
            *ptr++ = '\n';
            printed = 0;
        }
    }

    if ((i == derLen) && (printed != 0))
    {
        *ptr++ = '\n';
    }

    // 1 byte extra
    if (i == (derLen - 1))
    {
        tmp = der[i] << 4;
        *ptr++ = base64[(tmp >> 6) & 63];
        *ptr++ = base64[(tmp >> 0) & 63];
        *ptr++ = '=';
        *ptr++ = '=';
        *ptr++ = '\n';
    }

    // 2 byte extra
    if (i == (derLen - 2))
    {
        tmp = ((der[i] << 8) | (der[i + 1])) << 2;
        *ptr++ = base64[(tmp >> 12) & 63];
        *ptr++ = base64[(tmp >> 6) & 63];
        *ptr++ = base64[(tmp >> 0) & 63];
        *ptr++ = '=';
        *ptr++ = '\n';
    }

     portMemCopy(ptr, bufferLen - (ptr - buffer), SPDM_PEM_END_CERTIFICATE,
                 sizeof(SPDM_PEM_END_CERTIFICATE));
     ptr += sizeof(SPDM_PEM_END_CERTIFICATE);

    *bufferUsed = size;
    return NV_OK;
}

/*!
* Static function builds the cert chain in DER format. It is assumed that
* the all the certificates are valid. Also it is assumed that there is a valid
* spdm session already established.
*/
static NV_STATUS
_spdmBuildCertChainDer
(
    NvU8   *pFirstCert,
    NvU32   firstCertSize,
    NvU8   *pSecondCert,
    NvU32   secondCertSize,
    NvU8   *pOutBuffer,
    size_t *outBufferSize
)
{
    NvU64      remainingOutBufferSize = 0;
    NvU64      totalSize              = 0;
    void      *pPortMemCopyStatus     = NULL;

    if (pFirstCert == NULL || pSecondCert == NULL || pOutBuffer == NULL || outBufferSize == NULL)
    {
        return NV_ERR_INVALID_ARGUMENT;
    }

    // Calculate the total size of the certificate chain (in DER).
    totalSize = sizeof(SPDM_L1_CERTIFICATE_DER) +
                sizeof(SPDM_L2_CERTIFICATE_DER) +
                sizeof(SPDM_L3_CERTIFICATE_DER) +
                secondCertSize                  +
                firstCertSize;

    remainingOutBufferSize = *outBufferSize;
    if (remainingOutBufferSize < totalSize)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    //
    // Write the L1 DER certificate to the output buffer
    //
    pPortMemCopyStatus = portMemCopy(pOutBuffer,
                                     remainingOutBufferSize,
                                     SPDM_L1_CERTIFICATE_DER,
                                     sizeof(SPDM_L1_CERTIFICATE_DER));
    if (pPortMemCopyStatus == NULL)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    remainingOutBufferSize -= sizeof(SPDM_L1_CERTIFICATE_DER);
    pOutBuffer             += sizeof(SPDM_L1_CERTIFICATE_DER);

    //
    // Write the L2 DER certificate to the output buffer
    //
    pPortMemCopyStatus = portMemCopy(pOutBuffer,
                                     remainingOutBufferSize,
                                     SPDM_L2_CERTIFICATE_DER,
                                     sizeof(SPDM_L2_CERTIFICATE_DER));
    if (pPortMemCopyStatus == NULL)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    remainingOutBufferSize -= sizeof(SPDM_L2_CERTIFICATE_DER);
    pOutBuffer             += sizeof(SPDM_L2_CERTIFICATE_DER);

    //
    // Write the L3 DER certificate to the output buffer
    //
    pPortMemCopyStatus = portMemCopy(pOutBuffer,
                                     remainingOutBufferSize,
                                     SPDM_L3_CERTIFICATE_DER,
                                     sizeof(SPDM_L3_CERTIFICATE_DER));
    if (pPortMemCopyStatus == NULL)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    remainingOutBufferSize -= sizeof(SPDM_L3_CERTIFICATE_DER);
    pOutBuffer             += sizeof(SPDM_L3_CERTIFICATE_DER);

    //
    // Write the IK certificate in DER to the output buffer
    //
    pPortMemCopyStatus = portMemCopy(pOutBuffer,
                                     remainingOutBufferSize,
                                     pSecondCert,
                                     secondCertSize);
    if (pPortMemCopyStatus == NULL)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    remainingOutBufferSize -= secondCertSize;
    pOutBuffer             += secondCertSize;

    //
    // Write the AK certificate in DER to the output buffer
    //
    pPortMemCopyStatus = portMemCopy(pOutBuffer,
                                     remainingOutBufferSize,
                                     pFirstCert,
                                     firstCertSize);
    if (pPortMemCopyStatus == NULL)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    remainingOutBufferSize -= firstCertSize;
    pOutBuffer             += firstCertSize;

    // Output the total certificate chain size
    *outBufferSize = totalSize;

    return NV_OK;
}

/*!
* Static function that first converts the IK and AK certificates from DER to
* PEM format. Then it builds the cert chain in PEM format. It is assumed that
* the all the certificates are valid. Also it is assumed that there is a valid
* spdm session already established.
*/
static NV_STATUS
_spdmBuildCertChainPem
(
    NvU8   *pFirstCert,
    NvU32   firstCertSize,
    NvU8   *pSecondCert,
    NvU32   secondCertSize,
    NvU8   *pOutBuffer,
    size_t *outBufferSize
)
{
    NvU64              firstCertOutputSize      = 0;
    NvU64              secondCertOutputSize     = 0;
    NvU64              remainingOutBufferSize   = 0;
    void              *pPortMemCopyStatus       = NULL;
    NV_STATUS          status;

    if (pFirstCert == NULL || pSecondCert == NULL || pOutBuffer == NULL || outBufferSize == NULL)
    {
        return NV_ERR_INVALID_ARGUMENT;
    }

    remainingOutBufferSize = *outBufferSize;

    //
    // Write the AK certificate to the output buffer
    //
    status = pem_write_buffer(pFirstCert, firstCertSize, pOutBuffer,
                              remainingOutBufferSize, &firstCertOutputSize);
    if (status != NV_OK)
    {
        return status;
    }

    //
    // Keep track how much space we have left in the output buffer
    // and where the next certificate should start.
    // Clear the last byte (NULL).
    //
    remainingOutBufferSize -= firstCertOutputSize - 1;
    pOutBuffer             += firstCertOutputSize - 1;

    //
    // Write the IK certificate to the output buffer
    //
    status = pem_write_buffer(pSecondCert, secondCertSize, pOutBuffer,
                              remainingOutBufferSize, &secondCertOutputSize);
    if (status != NV_OK)
    {
        return status;
    }

    remainingOutBufferSize -= secondCertOutputSize - 1;
    pOutBuffer             += secondCertOutputSize - 1;

    // Checking if the available size of buffer is enough to keep the whole
    // certificate chain otherwise raise error.
    if (remainingOutBufferSize < sizeof(SPDM_L1_CERTIFICATE_PEM)
                               + sizeof(SPDM_L2_CERTIFICATE_PEM)
                               + sizeof(SPDM_L3_CERTIFICATE_PEM))
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    //
    // Write the L3 certificate to the output buffer
    //
    pPortMemCopyStatus = portMemCopy(pOutBuffer,
                                     remainingOutBufferSize,
                                     SPDM_L3_CERTIFICATE_PEM,
                                     sizeof(SPDM_L3_CERTIFICATE_PEM) - 1);
    if (pPortMemCopyStatus == NULL)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    remainingOutBufferSize -= sizeof(SPDM_L3_CERTIFICATE_PEM) - 1;
    pOutBuffer             += sizeof(SPDM_L3_CERTIFICATE_PEM) - 1;

    //
    // Write the L2 certificate to the output buffer
    //
    pPortMemCopyStatus = portMemCopy(pOutBuffer,
                                     remainingOutBufferSize,
                                     SPDM_L2_CERTIFICATE_PEM,
                                     sizeof(SPDM_L2_CERTIFICATE_PEM) - 1);
    if (pPortMemCopyStatus == NULL)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }
    remainingOutBufferSize -= sizeof(SPDM_L2_CERTIFICATE_PEM) - 1;
    pOutBuffer             += sizeof(SPDM_L2_CERTIFICATE_PEM) - 1;

    //
    // Write the L1 certificate to the output buffer
    //
    pPortMemCopyStatus = portMemCopy(pOutBuffer,
                                     remainingOutBufferSize,
                                     SPDM_L1_CERTIFICATE_PEM,
                                     sizeof(SPDM_L1_CERTIFICATE_PEM) - 1);
    if (pPortMemCopyStatus == NULL)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    //
    // Output the total certificate chain size
    // Do not count the NULL bytes.
    //
    *outBufferSize = firstCertOutputSize - 1 +
                     secondCertOutputSize - 1 +
                     sizeof(SPDM_L3_CERTIFICATE_PEM) - 1 +
                     sizeof(SPDM_L2_CERTIFICATE_PEM) - 1 +
                     sizeof(SPDM_L1_CERTIFICATE_PEM) - 1;

    return NV_OK;
}

/* ------------------------ Public Functions ------------------------------- */
NV_STATUS
spdmGetCertificates_GH100
(
    OBJGPU *pGpu,
    Spdm   *pSpdm
)
{
    NV_STATUS          status = NV_OK;
    NvU8              *pIkCertificate          = NULL;
    NvU32              ikCertificateSize       = 0;
    NvU8              *pAkCertificate          = NULL;
    NvU32              akCertificateSize       = 0;
    NvU8              *pGpuCerts               = NULL;
    size_t             gpuCertsSize            = 0;
    NvU8              *pDerCertChain           = NULL;
    size_t             derCertChainSize        = 0;
    NvU8              *pSpdmCertChainBufferEnd = NULL;
    libspdm_context_t *pContext                = NULL;
    uint32_t           base_hash_algo          = 0;
    NvU32              totalSize               = 0;

    if (pGpu == NULL || pSpdm == NULL)
    {
        return NV_ERR_INVALID_ARGUMENT;
    }

    if (pSpdm->pLibspdmContext == NULL)
    {
        return NV_ERR_NOT_READY;
    }

    // Allocate buffer for certificates.
    gpuCertsSize                    = LIBSPDM_MAX_CERT_CHAIN_SIZE;
    pGpuCerts                       = portMemAllocNonPaged(gpuCertsSize);
    derCertChainSize                = SPDM_MAX_ENCODED_CERT_CHAIN_SIZE;
    pDerCertChain                   = portMemAllocNonPaged(derCertChainSize);
    pSpdm->attestationCertChainSize = SPDM_MAX_ENCODED_CERT_CHAIN_SIZE;
    pSpdm->pAttestationCertChain    = portMemAllocNonPaged(pSpdm->attestationCertChainSize);

    // Ensure data was properly allocated.
    if (pGpuCerts == NULL || pDerCertChain == NULL || pSpdm->pAttestationCertChain == NULL)
    {
        status = NV_ERR_NO_MEMORY;
        goto ErrorExit;
    }

    portMemSet(pGpuCerts, 0, gpuCertsSize);
    portMemSet(pDerCertChain, 0, derCertChainSize);
    portMemSet(pSpdm->pAttestationCertChain, 0, pSpdm->attestationCertChainSize);

    // We fetch Attestation cert chain only on Hopper.
    CHECK_SPDM_STATUS(libspdm_get_certificate(pSpdm->pLibspdmContext, SPDM_CERT_DEFAULT_SLOT_ID,
                                              &gpuCertsSize, pGpuCerts));

    // Now, append the root certificates to create the entire chain.
    pContext = (libspdm_context_t *)pSpdm->pLibspdmContext;

    //
    // Skip over the certificate chain size, reserved size and the root hash
    // pSpdmCertChainBufferEnd represents last valid byte for cert buffer.
    //
    pSpdmCertChainBufferEnd = pGpuCerts + gpuCertsSize - 1;
    base_hash_algo          = pContext->connection_info.algorithm.base_hash_algo;
    pIkCertificate          = (NvU8 *)pGpuCerts;
    pIkCertificate         += sizeof(spdm_cert_chain_t) + libspdm_get_hash_size(base_hash_algo);

    // Calculate the size of the IK cert, and skip past it to get the AK cert.
    status = _calcX509CertSize(pIkCertificate, pSpdmCertChainBufferEnd, &ikCertificateSize);

    if (status != NV_OK)
    {
        goto ErrorExit;
    }

    pAkCertificate = (NvU8 *)pIkCertificate + ikCertificateSize;

    // Calculate the size of the AK certificate.
    status = _calcX509CertSize(pAkCertificate, pSpdmCertChainBufferEnd, &akCertificateSize);
    if (status != NV_OK)
    {
        return status;
    }

    // Make sure we have calculated the size correctly.
    if ((pAkCertificate + akCertificateSize - 1) != pSpdmCertChainBufferEnd)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    // Retrieve the entire certificate chain in DER format in order to validate it.
    status = _spdmBuildCertChainDer(pAkCertificate, akCertificateSize,
                                    pIkCertificate, ikCertificateSize,
                                    pDerCertChain,
                                    &derCertChainSize);

    if (status != NV_OK)
    {
        goto ErrorExit;
    }

    totalSize = sizeof(SPDM_L1_CERTIFICATE_DER)  +
                sizeof(SPDM_L2_CERTIFICATE_DER)  +
                sizeof(SPDM_L3_CERTIFICATE_DER)  +
                akCertificateSize                +
                ikCertificateSize;

    if (derCertChainSize != totalSize)
    {
        NV_PRINTF(LEVEL_ERROR, " %s: derCertChainSize(%lu) != totalSize(0x%x) !! \n",
                  __FUNCTION__, derCertChainSize, totalSize);

        // Something has gone quite wrong with our calculations.
        status = NV_ERR_BUFFER_TOO_SMALL;
        goto ErrorExit;
    }

    // Now, validate that the certificate chain is correctly signed.
    if (!libspdm_x509_verify_cert_chain(pDerCertChain, sizeof(SPDM_L1_CERTIFICATE_DER),
                                        pDerCertChain + sizeof(SPDM_L1_CERTIFICATE_DER),
                                        derCertChainSize - sizeof(SPDM_L1_CERTIFICATE_DER)))
    {
        status = NV_ERR_INVALID_DATA;
        goto ErrorExit;
    }

    //
    // Now that the cert chain is valid, retrieve the cert chain in PEM format,
    // as the Verifier can only parse PEM format.
    //
    status = _spdmBuildCertChainPem(pAkCertificate, akCertificateSize,
                                    pIkCertificate, ikCertificateSize,
                                    pSpdm->pAttestationCertChain,
                                    &pSpdm->attestationCertChainSize);
    if (status != NV_OK)
    {
        goto ErrorExit;
    }

ErrorExit:
    //
    // In both success and failure we need to free these allocated buffers.
    // portMemFree() will handle if they are NULL. On success, keep
    // the local pAttestationCertChain buffer.
    //
    portMemFree(pGpuCerts);
    portMemFree(pDerCertChain);

    if (status != NV_OK)
    {
        // portMemFree() handles NULL.
        portMemFree(pSpdm->pAttestationCertChain);
        pSpdm->pAttestationCertChain    = NULL;
        pSpdm->attestationCertChainSize = 0;
    }

    return status;
}

NV_STATUS
spdmGetCertChains_GH100
(
    OBJGPU *pGpu,
    Spdm   *pSpdm,
    void   *pKeyExCertChain,
    NvU32  *pKeyExCertChainSize,
    void   *pAttestationCertChain,
    NvU32  *pAttestationCertChainSize
)
{
    if (pGpu == NULL || pSpdm == NULL || pAttestationCertChain == NULL ||
        pAttestationCertChainSize == NULL)
    {
        return NV_ERR_INVALID_ARGUMENT;
    }

    // Check that we're in a valid state.
    if (pSpdm->pLibspdmContext == NULL || pSpdm->pAttestationCertChain == NULL ||
        pSpdm->attestationCertChainSize == 0)
    {
        return NV_ERR_NOT_READY;
    }

    // We only support Attestation certificates on Hopper.
    if (pKeyExCertChainSize != NULL)
    {
        pKeyExCertChainSize = 0;
    }

    if (*pAttestationCertChainSize < pSpdm->attestationCertChainSize)
    {
        return NV_ERR_BUFFER_TOO_SMALL;
    }

    portMemCopy(pAttestationCertChain, *pAttestationCertChainSize,
                pSpdm->pAttestationCertChain, pSpdm->attestationCertChainSize);
    *pAttestationCertChainSize = pSpdm->attestationCertChainSize;

    return NV_OK;
}
