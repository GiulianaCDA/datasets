1 #ifndef GPAC_DISABLE_ISOM
  2
  3 #ifndef GPAC_DISABLE_ISOM_HINTING
  4 #include <gpac/isomedia.h>
  5 #include <gpac/internal/isomedia_dev.h>
  6 #include <gpac/constants.h>
  7
  8 #ifndef GPAC_DISABLE_ODF
  9 #include <gpac/odf.h>
 10 #else
 11 #include <gpac/internal/odf_dev.h>
 12 #endif
 13
 14 #ifdef __cplusplus
 15 extern "C" {
 16 #endif
 17
 18 //encode an IOD from the ISOM file
 19 GF_EXPORT
 20 void gf_isom_write_iod(GF_ISOFile *file, Bool remove_ocr)
 21 {
 22 	u32 i;
 23 	GF_ESD *esd;
 24 	GF_IOD *iod;
 25 	char sdpLine[1024], buf64[2000];
 26 	unsigned char *buffer;
 27 	u32 size, size64;
 28
 29 	if (gf_isom_get_iod(file) ) return;
 30
 31 	iod = gf_odf_desc_new(GF_ODF_IOD_TAG);
 32 	gf_isom_set_iod(file, iod);
 33 	if (!gf_list_count(iod->ESDescriptors)) {
 34 		esd = gf_isom_get_esd(file, 0, 1);
 35 		if (esd) {
 36 #ifndef GPAC_DISABLE_SCENE_DESCRIPTION
 37 			//encode the scene description in Base64
 38 			if (esd->decoderConfig->streamType == GF_STREAM_SCENE) {
 39 				Bool is_ok = 1;
 40 				GF_ISOSample *samp;
 41 				gf_isom_get_sample(file, esd->ESID, 1, &samp);
 42 				if (samp && samp->dataLength) {
 43 					size64 = gf_base64_encode(samp->data, samp->dataLength, buf64, 2000);
 44 				} else {
 45 					is_ok = 0;
 46 				}
 47 #ifndef GPAC_DISABLE_RTP
 48 				sprintf(sdpLine, "mpeg4-esid:%d;mpeg4-channel=%d;mpeg4-profile-level=15;config=\"For4,%s\"", esd->ESID, esd->dependsOnESID ? 2 : 1, buf64);
 49 #else
 50                                 sprintf(sdpLine, "config=\"For4,%s\"", buf64);
 51 #endif
 52
 53 				if (esd->decoderConfig) {
 54 					esd->decoderConfig->avgBitrate = 0;
 55 					esd->decoderConfig->bufferSizeDB = samp->dataLength;
 56 					esd->decoderConfig->maxBitrate = 0;
 57 				}
 58 				esd->URLString = (char*)gf_malloc(sizeof(char) * (strlen(sdpLine)+1));
 59 				strcpy(esd->URLString, sdpLine);
 60 			} else {
 61 				GF_LOG(GF_LOG_ERROR, GF_LOG_CORE, ("[ISOMedia] Scene Description track found but no scene description sample\n"));
 62 				is_ok = 0;
 63 			}
 64 #endif /*GPAC_DISABLE_SCENE_DESCRIPTION*/
 65 			if (esd->decoderConfig && (esd->decoderConfig->streamType == GF_STREAM_OD)) {
 66 				gf_odf_desc_del((GF_Descriptor*) esd);
 67 				esd = NULL;
 68 			} else if (is_ok) {
 69 				gf_list_add(iod->ESDescriptors, esd);
 70 			}
 71 		}
 72 	}
 73 	for (i=0; i<gf_isom_get_track_count(file); i++) {
 74 		esd = gf_isom_get_esd(file, i+1, 1);
 75 		if (!esd) continue;
 76 		if (esd->decoderConfig && (esd->decoderConfig->streamType == GF_STREAM_OD)) {
 77 			gf_odf_desc_del((GF_Descriptor*) esd);
 78 		} else {
 79 			if (!gf_list_find(iod->ESDescriptors, esd) )
 80 				gf_list_add(iod->ESDescriptors, esd);
 81 		}
 82 	}
 83
 84 #ifndef GPAC_DISABLE_RTP
 85 	//build the SDP lines for each stream
 86 	if (gf_list_count(iod->ESDescriptors)) {
 87 		GF_ESD *esd;
 88 		u32 sdp_index = gf_isom_get_sdp_info(file, &esd);
 89 		if (esd) {
 90 			sprintf(sdpLine, "mpeg4-esid:%d;mpeg4-channel=%d;config=\"%s/%d\"", esd->ESID, esd->dependsOnESID ? 2 : 1, gf_4cc_to_str(esd->decoderConfig->objectTypeIndication), esd->slConfig->timestampResolution);
 91 			if (sdp_index) {
 92 				gf_isom_set_sdp_info(file, sdp_index, sdpLine, NULL, NULL);
 93 			} else {
 94 				gf_isom_add_sdp_info(file, "mp4", sdpLine, NULL, NULL);
 95 			}
 96 		}
 97 		i=0;
 98 		while ((esd = (GF_ESD *)gf_list_enum(iod->ESDescriptors, &i))) {
 99 			if (esd->dependsOnESID) continue;
100 			sprintf(sdpLine, "mpeg4-esid:%d;mpeg4-channel=%d", esd->ESID, esd->dependsOnESID ? 2 : 1);
101 			if (esd->decoderConfig) {
102 #ifndef GPAC_DISABLE_AV_PARSERS
103                         if (gf_isom_avc_config_get_pl_idc(esd->decoderConfig, NULL)) {
104 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=15");
105 				} else
106 #endif
107                         if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_MPEG4) {
108 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
109                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_AUDIO_AAC) {
110 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=15");
111 				} else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_AUDIO_MPEG2) {
112 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
113 				} else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_AUDIO_MPEG4) {
114 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
115                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_H263) {
116 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
117                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_H264) {
118 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
119                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_MJ2) {
120 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
121                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_AUDIO_SILK) {
122 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
123                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_VP8) {
124 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
125                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_VP9) {
126 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
127                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_AUDIO_AC3) {
128 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
129                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_AUDIO_EC3) {
130 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
131                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_HVC1) {
132 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
133                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_HEVC) {
134 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
135                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_AUDIO_OPUS) {
136 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
137                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_AV1) {
138 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
139                         } else if (esd->decoderConfig->objectTypeIndocation==GPAC_OTI_VIDEO_DIRAC) {
140 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
141                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_THEORA) {
142 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
143                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG) {
144 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
145                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_RAW) {
146 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
147                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_SVG) {
148 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
149                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_GIF) {
150 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
151                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_TIFF) {
152 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
153                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_PNG) {
154 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
155                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_BMP) {
156 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
157                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_TIFF) {
158 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
159                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_PNG) {
160 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
161                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_BMP) {
162 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
163                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
164 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
165                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
166 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
167                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
168 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
169                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
170 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
171                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
172 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
173                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
174 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
175                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
176 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
177                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
178 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
179                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
180 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
181                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
182 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
183                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
184 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
185                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
186 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
187                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
188 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
189                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
190 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
191                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
192 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
193                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
194 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
195                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
196 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
197                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
198 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
199                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
200 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
201                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
202 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
203                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
204 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
205                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
206 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
207                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
208 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
209                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
210 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
211                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
212 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
213                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
214 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
215                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
216 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
217                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
218 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
219                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
220 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
221                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
222 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
223                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
224 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
225                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
226 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
227                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
228 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
229                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
230 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
231                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
232 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
233                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
234 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
235                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
236 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
237                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
238 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
239                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
240 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
241                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
242 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
243                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
244 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
245                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
246 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
247                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
248 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
249                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
250 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
251                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
252 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
253                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
254 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
255                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
256 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
257                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
258 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
259                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
260 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
261                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
262 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
263                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
264 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
265                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
266 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
267                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
268 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
269                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
270 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
271                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
272 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
273                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
274 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
275                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
276 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
277                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
278 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
279                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
280 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
281                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
282 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
283                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
284 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
285                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
286 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
287                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
288 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
289                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
290 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
291                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
292 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
293                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
294 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
295                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
296 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
297                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
298 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
299                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
300 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
301                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
302 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
303                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
304 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
305                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
306 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
307                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
308 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
309                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
310 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
311                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
312 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
313                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
314 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
315                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
316 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
317                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
318 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
319                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
320 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
321                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
322 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
323                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
324 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
325                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
326 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
327                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
328 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
329                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
330 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
331                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
332 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
333                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
334 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
335                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
336 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
337                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
338 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
339                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
340 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
341                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
342 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
343                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
344 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
345                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
346 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
347                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
348 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
349                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
350 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
351                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
352 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
353                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
354 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
355                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
356 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
357                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
358 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
359                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
360 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
361                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
362 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
363                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
364 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
365                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
366 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
367                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
368 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
369                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
370 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
371                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
372 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
373                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
374 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
375                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
376 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
377                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
378 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
379                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
380 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
381                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
382 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
383                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
384 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
385                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
386 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
387                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
388 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
389                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
390 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
391                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
392 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
393                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
394 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
395                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
396 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
397                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
398 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
399                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
400 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
401                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
402 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
403                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
404 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
405                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
406 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
407                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
408 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
409                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
410 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
411                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
412 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
413                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
414 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
415                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
416 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
417                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
418 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
419                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
420 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
421                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
422 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
423                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
424 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
425                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
426 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
427                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
428 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
429                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
430 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
431                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
432 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
433                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
434 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
435                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
436 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
437                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
438 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
439                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
440 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
441                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
442 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
443                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
444 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
445                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
446 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
447                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
448 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
449                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
450 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
451                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
452 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
453                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
454 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
455                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
456 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
457                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
458 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
459                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
460 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
461                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
462 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
463                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
464 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
465                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
466 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
467                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
468 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
469                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
470 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
471                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
472 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
473                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
474 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
475                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
476 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
477                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
478 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
479                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
480 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
481                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
482 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
483                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
484 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
485                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
486 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
487                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
488 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
489                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
490 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
491                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
492 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
493                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
494 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
495                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
496 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
497                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
498 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
499                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
500 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
501                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
502 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
503                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
504 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
505                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
506 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
507                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
508 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
509                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
510 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
511                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
512 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
513                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
514 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
515                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
516 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
517                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
518 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
519                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
520 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
521                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
522 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
523                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
524 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
525                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
526 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
527                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
528 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
529                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
530 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
531                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
532 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
533                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
534 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
535                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
536 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
537                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
538 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
539                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
540 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
541                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
542 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
543                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
544 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
545                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
546 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
547                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
548 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
549                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
550 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
551                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
552 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
553                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
554 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
555                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
556 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
557                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
558 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
559                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
560 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
561                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
562 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
563                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
564 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
565                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
566 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
567                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
568 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
569                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
570 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
571                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
572 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
573                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
574 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
575                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
576 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
577                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
578 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
579                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
580 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
581                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
582 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
583                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
584 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
585                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
586 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
587                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
588 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
589                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
590 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
591                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
592 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
593                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
594 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
595                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
596 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
597                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
598 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
599                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
600 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
601                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
602 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
603                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
604 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
605                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
606 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
607                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
608 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
609                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
610 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
611                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
612 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
613                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
614 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
615                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
616 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
617                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
618 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
619                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
620 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
621                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
622 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
623                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
624 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
625                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
626 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
627                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
628 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
629                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
630 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
631                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
632 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
633                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
634 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
635                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
636 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
637                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
638 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
639                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
640 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
641                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
642 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
643                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
644 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
645                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
646 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
647                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
648 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
649                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
650 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
651                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
652 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
653                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
654 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
655                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
656 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
657                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
658 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
659                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
660 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
661                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
662 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
663                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
664 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
665                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
666 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
667                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
668 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
669                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
670 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
671                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
672 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
673                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
674 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
675                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
676 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
677                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
678 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
679                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
680 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
681                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
682 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
683                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
684 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
685                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
686 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
687                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
688 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
689                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
690 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
691                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
692 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
693                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
694 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
695                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
696 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
697                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
698 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
699                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
700 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
701                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
702 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
703                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
704 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
705                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
706 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
707                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
708 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
709                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
710 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
711                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
712 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
713                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
714 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
715                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
716 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
717                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
718 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
719                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
720 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
721                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
722 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
723                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
724 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
725                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
726 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
727                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
728 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
729                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
730 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
731                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
732 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
733                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
734 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
735                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
736 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
737                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
738 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
739                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
740 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
741                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
742 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
743                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
744 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
745                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
746 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
747                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
748 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
749                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
750 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
751                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
752 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
753                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
754 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
755                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
756 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
757                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
758 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
759                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
760 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
761                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
762 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
763                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
764 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
765                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
766 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
767                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
768 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
769                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
770 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
771                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
772 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
773                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
774 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
775                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
776 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
777                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
778 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
779                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
780 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
781                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
782 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
783                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
784 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
785                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
786 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
787                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
788 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
789                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
790 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
791                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
792 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
793                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
794 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
795                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
796 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
797                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
798 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
799                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
800 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
801                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
802 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
803                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
804 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
805                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
806 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
807                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
808 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
809                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
810 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
811                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
812 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
813                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
814 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
815                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
816 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
817                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
818 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
819                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
820 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
821                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
822 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
823                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
824 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
825                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
826 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
827                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
828 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
829                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
830 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
831                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
832 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
833                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
834 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
835                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
836 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
837                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
838 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
839                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
840 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
841                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
842 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
843                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
844 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
845                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
846 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
847                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
848 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
849                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
850 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
851                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
852 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
853                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
854 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
855                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
856 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
857                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
858 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
859                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
860 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
861                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
862 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
863                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
864 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
865                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
866 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
867                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
868 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
869                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
870 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
871                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
872 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
873                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
874 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
875                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
876 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
877                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
878 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
879                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
880 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
881                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
882 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
883                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
884 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
885                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
886 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
887                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
888 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
889                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
890 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
891                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
892 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
893                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
894 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
895                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
896 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
897                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
898 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
899                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
900 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
901                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
902 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
903                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
904 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
905                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
906 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
907                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
908 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
909                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
910 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
911                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
912 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
913                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
914 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
915                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
916 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
917                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
918 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
919                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
920 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
921                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
922 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
923                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
924 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
925                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
926 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
927                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
928 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
929                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
930 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
931                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
932 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
933                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
934 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
935                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
936 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
937                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
938 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
939                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
940 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
941                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
942 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
943                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
944 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
945                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
946 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
947                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
948 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
949                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
950 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
951                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
952 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
953                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
954 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
955                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
956 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
957                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
958 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
959                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
960 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
961                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
962 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
963                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
964 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
965                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
966 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
967                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
968 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
969                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
970 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
971                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
972 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
973                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
974 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
975                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
976 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
977                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
978 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
979                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
980 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
981                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
982 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
983                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
984 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
985                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
986 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
987                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
988 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
989                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
990 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
991                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
992 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
993                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
994 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
995                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
996 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
997                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
998 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
999                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1000 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1001                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1002 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1003                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1004 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1005                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1006 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1007                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1008 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1009                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1010 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1011                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1012 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1013                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1014 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1015                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1016 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1017                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1018 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1019                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1020 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1021                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1022 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1023                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1024 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1025                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1026 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1027                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1028 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1029                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1030 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1031                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1032 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1033                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1034 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1035                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1036 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1037                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1038 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1039                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1040 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1041                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1042 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1043                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1044 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1045                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1046 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1047                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1048 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1049                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1050 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1051                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1052 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1053                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1054 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1055                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1056 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1057                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1058 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1059                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1060 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1061                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1062 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1063                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1064 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1065                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1066 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1067                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1068 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1069                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1070 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1071                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1072 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1073                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1074 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1075                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1076 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1077                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1078 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1079                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1080 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1081                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1082 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1083                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1084 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1085                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1086 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1087                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1088 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1089                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1090 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1091                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1092 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1093                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1094 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1095                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1096 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1097                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1098 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1099                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1100 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1101                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1102 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1103                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1104 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1105                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1106 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1107                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1108 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1109                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1110 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1111                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1112 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1113                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1114 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1115                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1116 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1117                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1118 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1119                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1120 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1121                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1122 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1123                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1124 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1125                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1126 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1127                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1128 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1129                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1130 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1131                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1132 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1133                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1134 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1135                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1136 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1137                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1138 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1139                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1140 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1141                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1142 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1143                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1144 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1145                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1146 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1147                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1148 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1149                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1150 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1151                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1152 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1153                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1154 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1155                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1156 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1157                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1158 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1159                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1160 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1161                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1162 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1163                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1164 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1165                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1166 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1167                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1168 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1169                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1170 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1171                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1172 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1173                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1174 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1175                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1176 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1177                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1178 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1179                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1180 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1181                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1182 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1183                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1184 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1185                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1186 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1187                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1188 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1189                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1190 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1191                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1192 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1193                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1194 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1195                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1196 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1197                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1198 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1199                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1200 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1201                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1202 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1203                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1204 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1205                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1206 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1207                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1208 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1209                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1210 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1211                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1212 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1213                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1214 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1215                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1216 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1217                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1218 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1219                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1220 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1221                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1222 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1223                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1224 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1225                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1226 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1227                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1228 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1229                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1230 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1231                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1232 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1233                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1234 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1235                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1236 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1237                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1238 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1239                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1240 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1241                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1242 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1243                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1244 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1245                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1246 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1247                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1248 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1249                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1250 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1251                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1252 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1253                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1254 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1255                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1256 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1257                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1258 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1259                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1260 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1261                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1262 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1263                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1264 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1265                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1266 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1267                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1268 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1269                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1270 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1271                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1272 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1273                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEG2000) {
1274 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1275                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXS) {
1276 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1277                         } else if (esd->decoderConfig->objectTypeIndication==GPAC_OTI_VIDEO_JPEGXR) {
1278 					sprintf(sdpLine+strlen(sdpLine), ";mpeg4-profile-level=3");
1279                         } else if (esd->decoderClientType == GPAC_OTI_ESCAPED_JPEGXR) {
1280 					sprintf(szFileExt, ".jxr");
1281 				} else if (esd->decoderConfig && esd->decoderConfig->streamType == GPAC_STREAM_MPEG4) {
1282 					sprintf(szFileExt, "%s", ".mp4");
1283                 }
1284             }
1285         }
1286     } else if (od->objectDescriptorID == GPAC_OTI_ESD_ID) {
1287         ESD *esd = (ESD *) od;
1288 	    if (esd->decoderConfig && esd->decoderConfig->streamType == GPAC_STREAM_OD) {
1289             sprintf(szFileExt, "%s", ".ogg");
1290         } else if (esd->decoderConfig && esd->decoderConfig->streamType == GPAC_STREAM_MPEG4) {
1291 		    sprintf(szFileExt, "%s", ".mp4");
1292         }
1293     } else if (od->objectDescriptorID == GPAC_OTI_BIFS_PROG_ID) {
1294         sprintf(szFileExt, "%s", ".bifs");
1295     }
1296 }
1297 return szFileExt;
1298}
/* This function returns the file extension based on the object type and stream type of the input data.
The ObjectTypeIndication (OD) is used to determine the type of object, which can be a scene description,
movie description, audio stream, visual stream, or BIFS program. The function checks the objectDescriptorID
and decoderConfig fields of the OD structure to determine the appropriate file extension. If the objectDescriptorID
is GPAC_OTI_ESD_ID and the decoderConfig field indicates that the streamType is GPAC_STREAM_OD, the function returns ".ogg".
If the objectDescriptorID is GPAC_OTI_BIFS_PROG_ID, the function returns ".bifs". Otherwise, the function checks the values of the decoderClientType and decoderConfig fields to determine the appropriate file extension. */