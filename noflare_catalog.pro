; IDL CODE TO CREATE THE SET OF NON-FLARING ACTIVE-REGION FEATURES FOR FLARE PREDICTION
;-------------------------------------------------------------------------------------

; TO SAVE TIME AND REDUCE I/O, CREATE AN IDL BINARY FILE OF THE SHARP DATABASE 

;show_info ds="su_mbobra.sharp_cea_720s_clean[]" key=HARPNUM,T_REC,USFLUX,MEANGBT,MEANJZH,MEANPOT,SHRGT45,TOTUSJH,MEANGBH,MEANALP,MEANGAM,MEANGBZ,MEANJZD,TOTUSJZ,SAVNCPP,TOTPOT,MEANSHR,AREA_ACR,R_VALUE,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH,CRVAL1,CRLN_OBS,QUALITY -q > /tmp28/couvidat/sharp.txt
;show_info ds="su_mbobra.sharp_cea_720s_pil[]" key=HARPNUM,T_REC,USFLUX,MEANGBT,MEANJZH,MEANPOT,SHRGT45,TOTUSJH,MEANGBH,MEANALP,MEANGAM,MEANGBZ,MEANJZD,TOTUSJZ,SAVNCPP,TOTPOT,MEANSHR,AREA_ACR,R_VALUE,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH,CRVAL1,CRLN_OBS,QUALITY -q > /tmp28/couvidat/sharp_pil.txt
;idl
;OPENR,1,'/tmp28/couvidat/sharp.txt'
;num=1604937l
;;;num=1337657l
;;;;num=1335593l;for PIL
;usflux=FLTARR(num)
;harpnum=usflux
;TREC=STRARR(num)
;meangbt=usflux
;meanjzh=usflux
;meanpot=usflux
;shrgt45=usflux
;totusjh=usflux
;meangbh=usflux
;meanalp=usflux
;meangam=usflux
;meangbz=usflux
;meanjzd=usflux
;totusjz=usflux
;savncpp=usflux
;totpot=usflux
;meanshr=usflux
;area=usflux
;RVAL=usflux
;TOTFX=usflux
;TOTFY=usflux
;TOTFZ=usflux
;TOTBSQ=usflux
;EPSX=usflux
;EPSY=usflux
;EPSZ=usflux
;CRVAL1=usflux
;CRLNOBS=usflux
;QUALITY=LONARR(num)
;ABSNJZH=usflux
;the=1l
;FOR i=0l,num-1 DO BEGIN & line='' & READF,1,line & temp=STRSPLIT(line,/EXTRACT) &  harpnum[i]=FLOAT(temp[0]) & TREC[i]=temp[1] & usflux[i]=FLOAT(temp[2])& meangbt[i]=FLOAT(temp[3]) & meanjzh[i]=FLOAT(temp[4]) & meanpot[i]=FLOAT(temp[5]) & shrgt45[i]=FLOAT(temp[6]) & totusjh[i]=FLOAT(temp[7]) & meangbh[i]=FLOAT(temp[8]) & meanalp[i]=FLOAT(temp[9]) & meangam[i]=FLOAT(temp[10]) & meangbz[i]=FLOAT(temp[11]) & meanjzd[i]=FLOAT(temp[12]) & totusjz[i]=FLOAT(temp[13]) & savncpp[i]=FLOAT(temp[14]) & totpot[i] =FLOAT(temp[15]) & meanshr[i]=FLOAT(temp[16]) & area[i]=FLOAT(temp[17]) & RVAL[i]=FLOAT(temp[18]) & TOTFX[i]=FLOAT(temp[19]) & TOTFY[i]=FLOAT(temp[20]) & TOTFZ[i]=FLOAT(temp[21]) & TOTBSQ[i]=FLOAT(temp[22]) & EPSX[i]=FLOAT(temp[23]) & EPSY[i]=FLOAT(temp[24]) & EPSZ[i]=FLOAT(temp[25]) & ABSNJZH[i]=FLOAT(temp[26]) & CRVAL1[i]=FLOAT(temp[27]) & CRLNOBS[i]=FLOAT(temp[28]) & reads,temp[29],the,format='(Z)' & QUALITY[i]=the & ENDFOR
;SAVE,harpnum,trec,usflux,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH,CRVAL1,CRLNOBS,QUALITY,file='/tmp28/couvidat/sharp.bin'
;SAVE,harpnum,trec,usflux,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,CRVAL1,CRLNOBS,QUALITY,file='/tmp28/couvidat/sharp_pil.bin'


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


PRO flare_catalog2

RESTORE,'/tmp28/couvidat/sharp.bin'
;RESTORE,'/tmp28/couvidat/sharp_pil.bin' ; polarity inversion line
nelemsharp=N_ELEMENTS(AREA)
usfluxca=usflux
harpnumca=harpnum
TRECca=TREC
meangbtca=meangbt
meanjzhca=meanjzh
meanpotca=meanpot
shrgt45ca=shrgt45
totusjhca=totusjh
meangbhca=meangbh
meanalpca=meanalp
meangamca=meangam
meangbzca=meangbz
meanjzdca=meanjzd
totusjzca=totusjz
savncppca=savncpp
totpotca=totpot
meanshrca=meanshr
areaca=area
RVALca=RVAL
TOTFXca=TOTFX
TOTFYca=TOTFY
TOTFZca=TOTFZ
TOTBSQca=TOTBSQ
EPSXca=EPSX
EPSYca=EPSY
EPSZca=EPSZ
CRVAL1ca=CRVAL1
CRLNOBSca=CRLNOBS
QUALITYca=QUALITY
ABSNJZHa=ABSNJZH


;-------------------------------------------------------------
; CONVERT T_RECs INTO JULIAN DAYS FOR BOTH FLARE CATALOGS PROVIDED
; BY MONICA

nNOM=21;16;15;669
noM=STRARR(nNOM)
noMTREC=FLTARR(nNOM) ; WILL CONTAIN JULIAN DAYS OF NO FLARES 

;OPENR,1,'no-M-peak.txt'
OPENR,1,'no-M-peak_updated.txt'
READF,1,noM
CLOSE,1
FOR i=0,nNOM-1 DO BEGIN &  t=STRSPLIT(noM[i],' ',/EXTRACT) & noMTREC[i]=TIMECONVERT(t[1]) & ENDFOR

nYESM=490;484;454;4258
yesM=STRARR(nYESM)
yesMTREC=FLTARR(nYESM) ; WILL CONTAIN JULIAN DAYS OF FLARES
yesMHARP=LONARR(nYESM); WILL CONTAIN HARP NUMBERS OF FLARING ACTIVE REGIONS

;OPENR,1,'yes-M-peak.txt'
OPENR,1,'yes-M-peak_updated.txt'
READF,1,yesM
CLOSE,1
FOR i=0,nYESM-1 DO BEGIN &  t=STRSPLIT(yesM[i],' ',/EXTRACT) & yesMTREC[i]=TIMECONVERT(t[3]) & yesMHARP[i]=LONG(t[0]) & ENDFOR

;----------------------------------------------------------------
;PSQL command to find which HARP numbers exist (runs on n02)
SPAWN,'psql -h hmidb jsoc -c "SELECT harpnum FROM hmi.sharp_720s_shadow GROUP BY harpnum ORDER BY harpnum"',HARPNUM
nharps=N_ELEMENTS(HARPNUM)-4
harpnums=LONARR(nharps) ; list of HARPNUMs as longs
FOR i=0,nharps-1 DO harpnums[i]=LONG(HARPNUM[i+2])

;----------------------------------------------------------------
; BUILDS CATALOG OF NOs
;num=5000l
;num=5198l
num=5231l
usflux=FLTARR(num)
times=STRARR(num)
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
harpyo=LONARR(num)
TOTFX=usflux
TOTFY=usflux
TOTFZ=usflux
TOTBSQ=usflux
EPSX=usflux
EPSY=usflux
EPSZ=usflux
ABSNJZH=usflux
selected=LONARR(num)-1
; finds out which T_REC exist for this HARP number
seed=2l;1l
FOR i=0l,num-1l DO BEGIN 
    retry:
    ;harp=LONG(RANDOMU(seed)*(harpnums[nharps-1]-harpnums[0])+harpnums[0]) ; choose a HARPNUM
    ;SPAWN,'show_info key=T_FRST,T_LAST ds=su_mbobra.sharp_cea_720s_clean"['+STRTRIM(STRING(harp),1)+'][]" n=1 -q',TREC
    ;t=STRSPLIT(TREC,/EXTRACT)
    ;
    ;IF N_ELEMENTS(t) EQ 1 THEN GOTO,retry ; because not all HARP numbers exist

    ;TSTART=TIMECONVERT(t[0])
    ;TEND=TIMECONVERT(t[1])
    ;jultime=RANDOMU(seed)*(TEND-TSTART)+TSTART ; chooses a T_REC
    ;CALDAT,jultime,month,day,year,hour,minute,second
    ;TREC=STRTRIM(STRING(year),1)+'.'+STRTRIM(STRING(month),1)+'.'+STRTRIM(STRING(day),1)+'_'+STRTRIM(STRING(hour),1)+':'+STRTRIM(STRING(minute),1)+':'+STRTRIM(STRING(second),1)+'_UTC'

    selection=LONG(RANDOMU(seed)*nelemsharp)
    a=WHERE(selected EQ selection)
    IF (a[0] NE -1 OR QUALITYca[selection] GT 65536) THEN GOTO,retry ELSE selected[i]=selection

    flag=0

    harp=harpnumca[selection]
    jultime=TIMECONVERT(TRECca[selection])
    
    ; check that the couple HARPNUM+T_REC
    ; is not present in the yes flare catalog within 24h for T_REC
    a=WHERE(yesMHARP EQ harp)
    IF(a[0] NE -1) THEN BEGIN
        ;FOR ii=0,N_ELEMENTS(a)-1 DO IF (yesMTREC[a[ii]]-jultime LE 1.0 AND yesMTREC[a[ii]]-jultime GT 0.0) THEN flag=1; IF SELECTED T_REC IS WITHIN 1 DAY OF A FLARE IN THE SAME AR, DISCARD IT AND RETRY !!! ONLY IF FLARE IS AFTER T_REC
         FOR ii=0,N_ELEMENTS(a)-1 DO IF (ABS(jultime-yesMTREC[a[ii]]) LE 2.0) THEN flag=1 ; IF SELECTED T_REC IS WITHIN 2 DAYS OF A FLARE IN THE SAME AR, DISCARD IT AND RETRY
    ENDIF
    ; check that the T_REC is not present within 24h in the no flare catalog
    ; or 48h depending!
    ;a=WHERE(ABS(jultime-noMTREC) LE 1.0)
     a=WHERE(ABS(jultime-noMTREC) LE 2.0)
    IF(a[0] NE -1) THEN flag=1

    ; check that we are not at the limb
    ;SPAWN,'show_info ds=su_mbobra.sharp_cea_720s_clean"['+STRTRIM(STRING(harp),1)+']['+STRTRIM(TREC,1)+']" key=CRVAL1 -q',CRVAL
    ;SPAWN,'show_info ds=su_mbobra.sharp_cea_720s_clean"['+STRTRIM(STRING(harp),1)+']['+STRTRIM(TREC,1)+']" key=CRLN_OBS -q',CRLNOBS
    ;CRVAL=FLOAT(CRVAL)
    ;CRLNOBS=FLOAT(CRLNOBS)
    IF(ABS(CRVAL1ca[selection]-CRLNOBSca[selection]) GT 68.) THEN flag=1

    ;SPAWN,'show_info ds="su_mbobra.sharp_cea_720s_clean['+STRTRIM(STRING(harp),1)+']['+STRTRIM(TREC,1)+']" key=USFLUX,MEANGBT,MEANJZH,MEANPOT,SHRGT45,TOTUSJH,MEANGBH,MEANALP,MEANGAM,MEANGBZ,MEANJZD,TOTUSJZ,SAVNCPP,TOTPOT,MEANSHR,AREA_ACR,R_VALUE,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ -q',temp2

    ;temp=STRSPLIT(temp2,/EXTRACT)
        usflux[i]=usfluxca[selection];FLOAT(temp[0])
    ;IF temp[0] NE 0 THEN BEGIN
        meangbt[i]=meangbtca[selection];FLOAT(temp[1])
        meanjzh[i]=meanjzhca[selection];FLOAT(temp[2])
        meanpot[i]=meanpotca[selection];FLOAT(temp[3])
        shrgt45[i]=shrgt45ca[selection];FLOAT(temp[4])
        totusjh[i]=totusjhca[selection];FLOAT(temp[5])
        meangbh[i]=meangbhca[selection];FLOAT(temp[6])
        meanalp[i]=meanalpca[selection];FLOAT(temp[7])
        meangam[i]=meangamca[selection];=FLOAT(temp[8])
        meangbz[i]=meangbzca[selection];FLOAT(temp[9])
        meanjzd[i]=meanjzdca[selection];FLOAT(temp[10])
        totusjz[i]=totusjzca[selection];FLOAT(temp[11])
        savncpp[i]=savncppca[selection];FLOAT(temp[12])
        totpot[i] =totpotca[selection];FLOAT(temp[13])
        meanshr[i]=meanshrca[selection];FLOAT(temp[14])
        area[i]=areaca[selection];=FLOAT(temp[15])
        RVAL[i]=RVALca[selection];FLOAT(temp[16])
        TOTFX[i]=TOTFXca[selection];FLOAT(temp[17])
        TOTFY[i]=TOTFYca[selection];FLOAT(temp[18])
        TOTFZ[i]=TOTFZca[selection];FLOAT(temp[19])
        TOTBSQ[i]=TOTBSQca[selection];FLOAT(temp[20])
        EPSX[i]=EPSXca[selection];FLOAT(temp[21])
        EPSY[i]=EPSYca[selection];FLOAT(temp[22])
        EPSZ[i]=EPSZca[selection];FLOAT(temp[23])
        ABSNJZH[i]=ABSNJZHa[selection]
        harpyo[i]=harp
        times[i]=TRECca[selection]
    ;ENDIF ELSE flag=1
        if usflux[i] EQ 0.0 OR FINITE(meangbt[i]+meanjzh[i]+meanpot[i]+shrgt45[i]+totusjh[i]+meangbh[i]+meanalp[i]+meangam[i]+meangbz[i]+meanjzd[i]+totusjz[i]+savncpp[i]+totpot[i]+meanshr[i]+area[i]+RVAL[i]+TOTFX[i]+TOTFY[i]+TOTFZ[i]+TOTBSQ[i]+EPSX[i]+EPSY[i]+EPSZ[i]+ABSNJZH[i]) EQ 0 THEN flag=1

    ;IF flag EQ 1 THEN i=i-1 ELSE PRINT,selection,i,usflux[i],harp,TRECca[selection],RVAL[i],CRVAL1ca[selection]-CRLNOBSca[selection],TOTFX[i],TOTBSQ[i],EPSZ[i],format='(10e18.10)'
        IF flag EQ 1 THEN i=i-1 ELSE PRINT,i,'   ',harpyo[i],'  ',TRECca[selection]

ENDFOR    

SAVE,times,harpyo,file='noflare_catalog_48h_times.bin'
;SAVE,times,harpyo,file='noflare_catalog_48h_times_2l.bin'
;SAVE,times,harpyo,file='noflare_catalog_24h_times.bin'
SAVE,usflux,harpyo,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH,file='noflare_catalog_48h.bin'
;SAVE,usflux,harpyo,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH,file='noflare_catalog_48h_2l.bin'
;SAVE,usflux,harpyo,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,ABSNJZH,file='noflare_catalog_24h.bin'
;SAVE,usflux,harpyo,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,TOTFX,TOTFY,TOTFZ,TOTBSQ,EPSX,EPSY,EPSZ,file='noflare_catalog_24h_pil.bin'
;SAVE,usflux,harp,meangbt,meanjzh,meanpot,shrgt45,totusjh,meangbh,meanalp,meangam,meangbz,meanjzd,totusjz,savncpp,totpot,meanshr,area,RVAL,file='noflare_catalog_48h_3.bin'
;OPENW,1,'noflare_catalog_24h.txt'
;OPENW,1,'noflare_catalog_24h_pil.txt'
OPENW,1,'noflare_catalog_48h.txt'
;OPENW,1,'noflare_catalog_48h_2l.txt'
FOR i=0l,n_elements(usflux)-1l DO printf,1,usflux[i],meangbt[i],meanjzh[i],meanpot[i],shrgt45[i],totusjh[i],meangbh[i],meanalp[i],meangam[i],meangbz[i],meanjzd[i],totusjz[i],savncpp[i],totpot[i],meanshr[i],area[i],RVAL[i],TOTFX[i],TOTFY[i],TOTFZ[i],TOTBSQ[i],EPSX[i],EPSY[i],EPSZ[i],ABSNJZH[i],format='(25e18.10)'
CLOSE,1
READ,PAUSE

END
