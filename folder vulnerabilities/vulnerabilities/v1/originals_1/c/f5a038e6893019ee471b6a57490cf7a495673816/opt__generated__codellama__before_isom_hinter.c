  	gf_isom_set_hinting(file, GF_TRUE);
    gf_isom_setup_hinting(file);
  
    //add hint tracks
    for (i = 0; i < gf_isom_get_track_count(file); i++) {
      esd = gf_isom_get_esd(file, i + 1, 1);
      if (!esd) continue;
      if (esd->decoderConfig) {
        if (gf_isom_hint_track(file, i + 1)) {
          GF_LOG(GF_LOG_ERROR, GF_LOG_RTP, ("[rtp hinter] Error adding hint track for ESID %d\n", esd->ESID));
        } else {
          GF_LOG(GF_LOG_DEBUG, GF_LOG_RTP, ("[rtp hinter] Added hint track for ESID %d\n", esd->ESID));
        }
      }
    }
  