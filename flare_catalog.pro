; short un-optimized IDL program to process Monica's flare catalog
; using the SHARP parameters in the database
;-------------------------------------------------------------

FUNCTION TIMECONVERT,TREC
t=STRSPLIT(TREC,'_',/EXTRACT)
IF(N_ELEMENTS(t) GT 1) THEN BEGIN
    date=t[0]
    time=t[1]
    t=STRSPLIT(DATE,'.',/EXTRACT)
    year=LONG(t[0])
    month=LONG(t[1])
    day=LONG(t[2])
    t=STRSPLIT(TIME,':',/EXTRACT)
    hour=LONG(t[0])
    minute=LONG(t[1])
    second=LONG(t[2])
    RETURN,JULDAY(month,day,year,hour,minute,second) 
ENDIF ELSE RETURN,0.0

END

PRO flare_catalog

;OPENR,1,'/home/mbobra/pros/agu13/sharpkeys/matched_final.txt'
;SPAWN,'wc /home/mbobra/pros/agu13/sharpkeys/matched_final.txt',t
;OPENR,1,'/home/mbobra/pros/agu13/matched_final.txt'
;SPAWN,'wc /home/mbobra/pros/agu13/matched_final.txt',t
;OPENR,1,'yes-M-peak.txt'
;SPAWN,'wc yes-M-peak.txt',t
OPENR,1,'yes-M-peak_updated.txt'
SPAWN,'wc yes-M-peak_updated.txt',t

nelem=LONG(t)
nelem=nelem[0]
d=STRARR(nelem)
READF,1,d
CLOSE,1
usflux=FLTARR(nelem)
times=STRARR(nelem)
meangbt=usflux
meanjzh=usflux
meanpot=usflux
shrgt45=usflux
totusjh=usflux
meangbh=usflux
meanalp=usflux
meangam=usflux
meangbz=usflux
meanjzd=usflux
totusjz=usflux
savncpp=usflux
totpot=usflux
meanshr=usflux
area=usflux
RVAL=usflux
LOC=usflux
TOTFX=usflux
TOTFY=usflux
TOTFZ=usflux
TOTBSQ=usflux
EPSX=usflux
EPSY=usflux
EPSZ=usflux
ABSNJZH=usflux
HARP=LONARR(nelem)
JULRECFLARE=FLTARR(nelem)
class=STRARR(Nelem)

FOR i=0,nelem-1 DO BEGIN
    temp=STRSPLIT(d[i],/EXTRACT)
    class[i]=temp[2]
    TREC=TIMECONVERT(temp[3]) ;TIME AT WHICH THE FLARE PEAKED
   ;TREC -= 0.5;1.0 ;SUBTRACT 1 DAY OR HALF A DAY
    TREC -= 1.0 ;SUBTRACT 1 DAY OR HALF A DAY
    CALDAT,TREC,month,day,year,hour,minute,second
    TREC=STRTRIM(STRING(year),1)+'.'+STRTRIM(STRING(month),1)+'.'+STRTRIM(STRING(day),1)+'_'+STRTRIM(STRING(hour),1)+':'+STRTRIM(STRING(minute),1)+':'+STRTRIM(STRING(second),1)+'_UTC'
    text='show_info ds="su_mbobra.sharp_cea_720s_clean_v2['+temp[0]+']['+TREC+']" key=USFLUX,MEANGBT,MEANJZH,MEANPOT,SHRGT45,TOTUSJH,MEANGBH,MEANALP,MEANGAM,MEANGBZ,MEANJZD,TOTUSJZ,SAVNCPP,TOTPOT,MEANSHR,AREA_ACR,R_VALUE,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH -q'
    SPAWN,'show_info ds="su_mbobra.sharp_cea_720s_clean_v2['+temp[0]+']['+TREC+']" key=T_REC -q',TREC
    IF(long(TREC) NE 0) THEN BEGIN
        SPAWN,'show_info ds="su_mbobra.sharp_cea_720s_clean_v2['+temp[0]+']['+TREC+']" key=CRVAL1 -q',CRVAL
        SPAWN,'show_info ds="su_mbobra.sharp_cea_720s_clean_v2['+temp[0]+']['+TREC+']" key=CRLN_OBS -q',CRLNOBS
        CRVAL=FLOAT(CRVAL)
        CRLNOBS=FLOAT(CRLNOBS)
        HARP[i]=LONG(temp[0])
        SPAWN,text,temp2
        temp=STRSPLIT(temp2,/EXTRACT)
        usflux[i]=FLOAT(temp[0])
        IF (temp[0] NE 0) AND (ABS(CRVAL-CRLNOBS) LE 68.0) THEN BEGIN ; we only select AR with a center to limb angle lower than 70 degrees
            meangbt[i]=FLOAT(temp[1])
            meanjzh[i]=FLOAT(temp[2])
            meanpot[i]=FLOAT(temp[3])
            shrgt45[i]=FLOAT(temp[4])
            totusjh[i]=FLOAT(temp[5])
            meangbh[i]=FLOAT(temp[6])
            meanalp[i]=FLOAT(temp[7])
            meangam[i]=FLOAT(temp[8])
            meangbz[i]=FLOAT(temp[9])
            meanjzd[i]=FLOAT(temp[10])
            totusjz[i]=FLOAT(temp[11])
            savncpp[i]=FLOAT(temp[12])
            totpot[i] =FLOAT(temp[13])
            meanshr[i]=FLOAT(temp[14])
            area[i]=FLOAT(temp[15])
            RVAL[i]=FLOAT(temp[16])
            TOTFX[i]=FLOAT(temp[17])
            TOTFY[i]=FLOAT(temp[18])
            TOTFZ[i]=FLOAT(temp[19])
            TOTBSQ[i]=FLOAT(temp[20])
            EPSX[i]=FLOAT(temp[21])
            EPSY[i]=FLOAT(temp[22])
            EPSZ[i]=FLOAT(temp[23])
            ABSNJZH[i]=FLOAT(temp[24])
            LOC[i]=CRVAL-CRLNOBS
            times[i]=TREC
        ENDIF ELSE usflux[i]=0.0
        JULRECFLARE[i]=TIMECONVERT(TREC)
    ENDIF
    PRINT,i,usflux[i],HARP[i],TREC,JULRECFLARE[i],RVAL[i],LOC[i],format='(7e18.10)'
ENDFOR

a=WHERE(usflux NE 0.0)
usflux=usflux[a]
harp=harp[a]
meangbt=meangbt[a]
meanjzh=meanjzh[a]
meanpot=meanpot[a]
shrgt45=shrgt45[a]
totusjh=totusjh[a]
meangbh=meangbh[a]
meanalp=meanalp[a]
meangam=meangam[a]
meangbz=meangbz[a]
meanjzd=meanjzd[a]
totusjz=totusjz[a]
savncpp=savncpp[a]
 totpot=totpot[a]
meanshr=meanshr[a]
area=area[a]
julrecflare=julrecflare[a]
RVAL=RVAL[a]
LOC=LOC[a]
TOTFX=TOTFX[a]
TOTFY=TOTFY[a]
TOTFZ=TOTFZ[a]
TOTBSQ=TOTBSQ[a]
EPSX=EPSX[a]
EPSY=EPSY[a]
EPSZ=EPSZ[a]
ABSNJZH=ABSNJZH[a]
class=class[a]
times=times[a]

; SAVE RESULTS IN IDL BINARY AND TEXT FILES

;SAVE,times,harp,file='flare_catalog_24h_times.bin'
SAVE,times,harp,file='flare_catalog_24h_times_test.bin'
;SAVE,usflux,harp,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meangbh,meanjzd,totusjz,savncpp,totpot,meanshr,usflux2,harp2,meangbt2,meanjzh2,meanpot2,shrgt452,totusjh2,meangbh2,meanalp2,meangam2,meangbz2,meangbh2,meanjzd2,totusjz2,savncpp2,totpot2,meanshr2,file='flare_catalog.bin'
;SAVE,usflux,harp,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,LOC,file='flare_catalog_24h.bin'
;SAVE,usflux,harp,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,LOC,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,file='flare_catalog_12h.bin'
;SAVE,usflux,harp,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,LOC,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH,class,file='flare_catalog_24h.bin'
SAVE,usflux,harp,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,LOC,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH,class,file='flare_catalog_24h_test.bin'
;OPENW,1,'flare_catalog.txt'
;OPENW,1,'flare_catalog_24h.txt'
 OPENW,1,'flare_catalog_24h_test.txt'
;OPENW,1,'flare_catalog_24h_pil.txt'
;OPENW,1,'flare_catalog_12h.txt'
FOR i=0,n_elements(usflux)-1 DO printf,1,usflux[i],meangbt[i],meanjzh[i],meanpot[i],shrgt45[i],totusjh[i],meangbh[i],meanalp[i],meangam[i],meangbz[i],meanjzd[i],totusjz[i],savncpp[i],totpot[i],meanshr[i],area[i],RVAL[i],TOTFX[i],TOTFY[i],TOTFZ[i],TOTBSQ[i],EPSX[i],EPSY[i],EPSZ[i],ABSNJZH[i],format='(25e18.10)'

CLOSE,1


END
