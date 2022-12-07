##########################################################
# aim of this script is to have a class                  #
# that gives you the dictionary of the binary file       #
##########################################################
import datetime
import collections
import numpy as np
import os
import zlib
import zipfile

class convertion():
    def __init__(self, filename, keep_threshold=False):
        self.filename = filename # these are the zip files
        self.keep_threshold = keep_threshold
     
    def read_raw(self):
        DetectInFile  = 0
        DetectTotal   = 0
        NbErrors      = 0
        data_vec      = []
        self.serial   = 0 
        self.version  = 0
        
        with zipfile.ZipFile(self.filename) as z:
            for filename in z.namelist():
                if (not os.path.isdir(filename)) and (z.getinfo(filename).file_size > 0):
                     f = z.open(filename, "r")  
                else:       
                     return data_vec, self.serial, self.version
            
        while(True):
            ReadByte = f.read(4)
            if ReadByte == b"":
                DetectInFile = 0
                break

            HeaderFixedCode = int.from_bytes(ReadByte, byteorder='big')
            if HeaderFixedCode != 0x22d5ebe7 :
                self.logger.warning("*** Error header fixed code: %s - 0x22d5ebe7" % hex(HeaderFixedCode))
                NbErrors += 1
                break
            DetectInFile += 1
            DetectTotal  += 1

            #### General header
            self.serial     = int.from_bytes(f.read(4), byteorder = 'big')
            self.version    = int.from_bytes(f.read(4), byteorder = 'big')
            UnixTimeSeconds = int.from_bytes(f.read(4), byteorder = 'big')
            UnixTimeMs      = int.from_bytes(f.read(4), byteorder = 'big')
            TimeStampConv   = datetime.datetime.fromtimestamp(UnixTimeSeconds).strftime('%Y-%m-%d %H:%M:%S')
            self.timestamp  = TimeStampConv + str(UnixTimeMs)  # to be changed
            NumberOfModules = int.from_bytes(f.read(4), byteorder = 'big')
            f.read(12)


            #### Scattering header
            LT11Framelength    = int.from_bytes(f.read(4), byteorder = 'big')
            EmitterID          = int.from_bytes(f.read(2), byteorder = 'big')
            DetectionID        = int.from_bytes(f.read(2), byteorder = 'big')
            ImageSize          = int.from_bytes(f.read(4), byteorder = 'big')
            f.read(20)


            if NumberOfModules == 3:
                #### FluoHeader
                LT03Framelength    = int.from_bytes(f.read(4), byteorder = 'big')
                EmitterID          = int.from_bytes(f.read(2), byteorder = 'big')
                DetectionID        = int.from_bytes(f.read(2), byteorder = 'big')
                f.read(24)
                if LT03Framelength != 256:
                    self.logger.warning ("***ERROR LT3 Header -- Wrong Frame length: %d" %(LT03Framelength))
                    NbErrors +=1

                #### LifeTimeHeader
                LT04Framelength    = int.from_bytes(f.read(4), byteorder = 'big')
                EmitterID          = int.from_bytes(f.read(2), byteorder = 'big')
                DetectionID        = int.from_bytes(f.read(2), byteorder = 'big')
                f.read(24)
                if LT04Framelength != 128:
                    self.logger.warning ("***ERROR LT4 Header -- Wrong Frame length: %d" %(LT04Framelength))
                    NbErrors += 1


            #### Scattering Image, uint32, big-endian
            ImageRaw = f.read(ImageSize*4)
            self.scat_image = np.fromstring(ImageRaw, dtype=np.dtype('>u4')).tolist()
            CRC = zlib.crc32(ImageRaw)

            #### Scattering Threshold, uint32 , big-endian
            Thresholds = f.read((LT11Framelength -ImageSize)*4)
            self.thresholds = np.fromstring(Thresholds, dtype=np.dtype('>u4')).tolist()
            CRC = zlib.crc32(Thresholds, CRC)


            if NumberOfModules == 3:
                #### Fluo Image, int32, big endian
                FluoRaw = f.read(LT03Framelength*4)
                self.spect_image = np.fromstring(FluoRaw, dtype=np.dtype('>i4')).tolist()
                CRC = zlib.crc32(FluoRaw, CRC)

                #### Lifetime Image, int16, big endian
                LifeRaw = f.read(LT04Framelength*4)
                self.life_image = np.fromstring(LifeRaw, dtype=np.dtype('>i2')).tolist()
                CRC = zlib.crc32(LifeRaw, CRC)


            #### Footer
            FooterCRC = int.from_bytes(f.read(4),byteorder='big')
            if CRC != FooterCRC:
                self.logger.warning("***ERROR CRC do not match: %s - %s" %(hex(FooterCRC), hex(CRC)))
                NbErrors += 1
                break

            f.read(124)

            #### check correct footer final code
            FooterFixedCode = int.from_bytes(f.read(4), byteorder = 'big')
            if FooterFixedCode != 0xf82f5be4:
                self.logger.warning("***Error footer fixed code: %s - 0xf82f5be4" % hex(FooterFixedCode))
                NbErrors += 1

            #### appending particle to DIC
            data_vec.append(self.particles_to_dic(NumberOfModules))

        return data_vec, self.serial, self.version

    def global_dic(self):
        data_vec, serial, version = self.read_raw()
        DIC = collections.defaultdict(list)
        DIC['Header']={}
        DIC['Header']['Device']={}
        DIC['Header']['Device']['Serial'] = serial
        DIC['Header']['Device']['Version'] = version
        DIC['Header']['Alignment'] = {}
        DIC['Header']['Alignment']['Scattering angles'] = np.linspace(-45,45,24).tolist()
        DIC['Header']['Alignment']['Spectral wavelengths'] = np.linspace(350,800,32).tolist()
        DIC['Header']['Alignment']['Lifetime ranges'] = [[350,400], [420, 460], [511,572], [672, 800]]
        DIC['Header']['Units']= {}
        DIC['Header']['Units']['Lifetime']= ['Time, ns', 'Amplitude, NA']
        DIC['Header']['Units']['Spectrometer'] = ['Wavelength, nm', 'Amplitude, NA']
        DIC['Header']['Units']['Scattering'] = ['Time, us', 'Angle, deg', 'Amplitude, NA']
        DIC['Data'] = data_vec
        return DIC

    def particles_to_dic(self, NumberOfModules):
        GDIC                        = collections.defaultdict(list)
        GDIC['Timestamp']           = self.timestamp
        GDIC['Scattering']          = {}
        GDIC['Scattering']['Image'] = self.scat_image
        if self.keep_threshold: # write the scat thresholds if keep_thresholds true
            GDIC['Scattering']['Thresholds'] = self.thresholds
        if NumberOfModules == 3:
            GDIC['Spectrometer']        = self.spect_image
            GDIC['Lifetime']            = self.life_image
        return GDIC