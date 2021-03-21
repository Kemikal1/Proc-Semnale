import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import ly as ly
def sine (amplitude, frequency, time, phase):
    return amplitude * np.sin (2 * np.pi * frequency * time + phase)
def tone_time(start,sampling_period,time):
    return np.linspace(start,start+time,int(time/sampling_period+1))
def ritm(bpm):
    whole = 60 / bpm
    half = whole / 2
    quarter = half / 2
    eight = quarter / 2
    sixteenth = eight / 2
    return whole,half,quarter,eight,sixteenth


bpm=60
whole,half,quarter,eight,sixteenth=ritm(bpm)


time_of_view = 1    # s
frequency = 440      # Hz
amplitude = 10000
phase = 0

sampling_rate = 44100
sampling_period = 1./sampling_rate  # s


n_samples = time_of_view/sampling_period
time = np.linspace (0, time_of_view, int(n_samples + 1))
tone = sine(amplitude, frequency, time, phase)
sd.default.samplerate = sampling_rate

time1= tone_time(time_of_view,sampling_period,whole)

wav_wave = np.array(tone, dtype=np.int16)
tone1=sine(amplitude, frequency*2, time1, phase)
wav_wave=np.append(wav_wave,tone1)
wav_wave = np.array(wav_wave, dtype=np.int16)
print(time1)
print(tone1)
print("\n")
print(time)
print(tone)

#print(wav_wave1[44000:44111])



"Ex1"
Do=523
Re=587
Mi=659
Fa=698
Sol=783
La=880
Si=987
Do1=1046
labels=["Do","Re","Mi","Fa","Sol","La","Si","Do"]
labels1=["c","d","e","f","g","a","b"]
gama=[Do,Re,Mi,Fa,Sol,La,Si,Do1]


duratel=[1,2,4,8,16]

t=tone_time(0,sampling_period,half)

wav=sine(amplitude,Do,t,0)

fig, axs = plt.subplots(8)
j=0
axs[j].plot(t[:int(len(wav)/50)],wav[:int(len(wav)/50)],label=labels[j])
axs[j].legend()

for i in gama[1:]:
    j+=1
    t=tone_time(0,sampling_period,half)
    fq=i
    nota=sine(amplitude,fq,t,phase)
    wav=np.append(wav[:int(wav.size)-100],nota[100:])
    axs[j].plot(t[:int(len(wav) / 50)], nota[:int(len(wav) / 50)],label="Nota "+labels[j])
    axs[j].legend()

plt.show()
wav = np.array(wav, dtype=np.int16)
#sd.play(wav, blocking=True)
sd.stop()



"Ex2"
note=[329,310,280,310,262,280,310,370,329,310,329,280,262,280,262,206]
durata=[2*whole,whole+half,whole-quarter-eight,whole-quarter-eight,half,quarter,4*whole-half,
        whole+half,half+eight+sixteenth,whole+quarter,whole-quarter-eight,whole-quarter-eight,half,quarter,2*whole+quarter,2*whole+half]
t=tone_time(0,sampling_period,4*half)
print("\n")

print("\n")
print(len(note))

wav=sine(amplitude,329,t,0)
for i in range(len(note)-1):

    t=tone_time(0,sampling_period,durata[i+1])
    fq=note[i+1]
    nota = sine(amplitude, fq, t, phase)
    wav = np.append(wav[:int(wav.size)], nota[:])
wav = np.array(wav, dtype=np.int16)
sd.play(wav, blocking=True)
sd.stop()


"Ex3"
s=0

t=tone_time(0,sampling_period,eight)
wav=sine(amplitude,1,t,0)
start=eight
with open('test.ly', 'r') as file:
    for line in file:
        for word in line.split():
            if word=="}":
                s-=2
            if s == 2 and (len(word)<4):
                d=0
                if word[0]=="r":
                    for i in word[1:]:
                        d*=10
                        d+=int(i)
                    durata=d
                    fq=1
                    if d!=0:
                        durata = d
                else:
                    for i in word[1:]:
                        d *= 10
                        d += int(i)
                    if d!=0:
                        durata = d
                    fq=gama[labels1.index(word[0])]
                t = tone_time(start, sampling_period, durate[duratel.index(durata)])
                nota = sine(amplitude, fq, t, phase)
                wav = np.append(wav[:int(wav.size)-10], nota[:])
                start+=durata

            if s==3:
                t=0
                for i in word[2:]:
                    t*=10
                    t+=int(i)
                bpm=t
                whole, half, quarter, eigth, sixteenth = ritm(bpm)
                durate = [whole, half, quarter, eight, sixteenth]
                s=2

            if word=="\\"+"tempo":
                s=3

            if word=="\\"+"relative":
                s=1
            if word=="{":
                s+=1
wav = np.array(wav, dtype=np.int16)
#sd.play(wav, blocking=True)

