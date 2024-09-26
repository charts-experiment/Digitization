import sys, time, struct
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import casperfpga

acc_len = 100 #2*(2**28)//4096

def get_vacc_data(fpga, bram_name, nchannels=1, nfft=1024):
    acc_n = fpga.read_uint('acc_cnt')
    chunk = nfft // nchannels
    raw = np.zeros((nchannels, chunk))
    raw = struct.unpack('>{:d}Q'.format(chunk), fpga.read(bram_name, chunk * 8, 0))
    return acc_n, np.array(raw)

def plot_spectrum(fpga, bram_names, cx=True, num_acc_updates=None):
    fig, axs = plt.subplots(4, 2, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    fs = (3932.16/2) / 8
    if cx:
        print('complex design')
        Nfft = 2**10
        fbins0 = np.arange(-Nfft//2, Nfft//2)
        fbins1 = np.arange(-(Nfft//2), 2*(Nfft//2))
        fbins2 = np.arange(-2*(Nfft//2), 3*(Nfft//2))
        fbins3 = np.arange(-3*(Nfft//2), 4*(Nfft//2))
        fbins4 = np.arange(-4*(Nfft//2), 5*(Nfft//2))
        fbins5 = np.arange(-5*(Nfft//2), 6*(Nfft//2))
        fbins6 = np.arange(-6*(Nfft//2), 7*(Nfft//2))
        fbins7 = np.arange(-7*(Nfft//2), 8*(Nfft//2))
        fbins_list = [fbins0, fbins1, fbins2, fbins3, fbins4, fbins5, fbins6, fbins7]
        df = fs / Nfft
        nchannels = 1
    else:
        print('real design')
        Nfft = 2**10
        #fs /= 2
        fbins0 = np.arange(0, Nfft // 2)
        fbins1 = np.arange((Nfft // 2), 2*(Nfft // 2))
        fbins2 = np.arange(2*(Nfft // 2), 3*(Nfft // 2))
        fbins3 = np.arange(3*(Nfft // 2), 4*(Nfft // 2))
        fbins4 = np.arange(4*(Nfft // 2), 5*(Nfft // 2))
        fbins5 = np.arange(5*(Nfft // 2), 6*(Nfft // 2))
        fbins6 = np.arange(6*(Nfft // 2), 7*(Nfft // 2))
        fbins7 = np.arange(7*(Nfft // 2), 8*(Nfft // 2))
        fbins_list = [fbins0, fbins1, fbins2, fbins3, fbins4, fbins5, fbins6, fbins7]
        df = fs / Nfft
        nchannels = 1
        
        

    full_scale = 2**14  # Assuming a 14-bit ADC
    # Define x-axis labels and limits
    x_limits_real = [
        (0, 122.88),
        (122.88, 245.76),
        (245.76, 368.64),
        (368.64, 491.52),
        (491.52, 614.4),
        (614.4, 737.28),
        (737.28, 860.16),
        (860.16, 983.04)
    ]
    x_limits_cx = [
        (0, 245.76),
        (245.76, 491.52),
        (491.52, 737.28),
        (737.28, 983.04),
        (983.04, 1228.8),
        (1228.8, 1474.56),
        (1474.56, 1720.32),
        (1720.32, 1966.08)
    ]
    

    lines = []

    for i, bram_name in enumerate(bram_names):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        fbins = fbins_list[i]
        if cx:
            acc_n, spectrum = get_vacc_data(fpga, bram_name, nchannels=nchannels, nfft=Nfft)
            spectrum = np.abs(fft.fftshift(spectrum)) / Nfft
            #fbins = fbins[:len(spectrum)]
            print(1)
        else:
            acc_n, spectrum = get_vacc_data(fpga, bram_name, nchannels=nchannels, nfft=Nfft // 2)
            spectrum = np.abs(spectrum) / (Nfft // 2)
            #spectrum = spectrum /full_scale
            fbins = fbins[:len(spectrum)]
                
        spectrum_dBFS = 10 * np.log10(spectrum + 1)
        faxis = fbins * df
        line, = ax.plot(faxis, spectrum_dBFS, '-')
        lines.append(line)

        ax.set_ylim(-10, 120)
        if cx:
           ax.set_xlim(x_limits_cx[i][0], x_limits_cx[i][1])  # Always plot the full range
           #ax.set_xticks(np.linspace(0, 122.88, 5))  # Set tick marks
           ax.set_xlabel(f'{x_limits_cx[i][0]} to {x_limits_cx[i][1]} MHz')
        else:
           ax.set_xlim(x_limits_real[i][0], x_limits_real[i][1])  # Always plot the full range
           #ax.set_xticks(np.linspace(0, 122.88, 5))  # Set tick marks
           ax.set_xlabel(f'{x_limits_real[i][0]} to {x_limits_real[i][1]} MHz')
           
        ax.grid()

    # Set a single title for the entire figure
    fig.suptitle('Accumulation Number: 0', fontsize=16)

    def update(frame, *fargs):
        #acc_n = None
        num_points = Nfft // 2  # For real design, adjust this as needed
        num_brams = len(bram_names)

        spectra = np.zeros((num_points, num_brams))
        for i, bram_name in enumerate(bram_names):
            print(bram_name)
            row = i // 2
            col = i % 2
            ax = axs[row, col]

            if cx:
                acc_n, spectrum = get_vacc_data(fpga, bram_name, nchannels=nchannels, nfft=Nfft)
                spectrum = np.abs(fft.fftshift(spectrum)) / Nfft
            else:
                acc_n, spectrum = get_vacc_data(fpga, bram_name, nchannels=nchannels, nfft=Nfft // 2)
                spectrum = np.abs(spectrum) / (Nfft // 2)
                #spectrum = spectrum /full_scale
                spectra[:,i] = spectrum
            
            if i in range(1, 8, 2):
                spectrum = spectrum[::-1]
            else:
                spectrum = spectrum    
            
            spectrum_dBFS = 10 * np.log10(spectrum + 1)
            print(max(spectrum_dBFS))
            print(min(spectrum_dBFS))
            print('DR:', max(spectrum_dBFS)-min(spectrum_dBFS))    
            lines[i].set_ydata(spectrum_dBFS)
            #lines[i].set_xdata(frequencies{:d})

        # Update the figure title with the current accumulation number
        fig.suptitle(f'Accumulation Number: {acc_n}', fontsize=16)
        diff_spectra = np.diff(spectra, axis=1)
        print(diff_spectra) 
    v = anim.FuncAnimation(fig, update, frames=1, repeat=True, fargs=None, interval=1000)
    plt.show()

if __name__ == "__main__":
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('rfsoc4x2_tut_spec_dbfs.py <HOSTNAME_or_IP> cx|real [options]')
    p.set_description(__doc__)
    p.add_option('-l', '--acc_len', dest='acc_len', type='int', default=2 * (2**28) // 2048,
                 help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048')
    p.add_option('-s', '--skip', dest='skip', action='store_true',
                 help='Skip programming and begin to plot data')
    p.add_option('-b', '--fpg', dest='fpgfile', type='str', default='',
                 help='Specify the fpg file to load')
    p.add_option('-a', '--adc', dest='adc_chan_sel', type=int, default=0,
                 help='adc input to select values are 0, 1, 2, or 3')

    opts, args = p.parse_args(sys.argv[1:])
    if len(args) < 2:
        print('Specify a hostname or IP for your casper platform. And either cx|real to indicate the type of spectrometer design.\n'
              'Run with the -h flag to see all options.')
        exit()
    else:
        hostname = args[0]
        mode_str = args[1]
        if mode_str == 'cx':
            mode = 1
        elif mode_str == 'real':
            mode = 0
        else:
            print('operation mode not recognized, must be "cx" or "real"')
            exit()

    if opts.fpgfile != '':
        bitstream = opts.fpgfile
    else:
        if mode == 1:
            fpg_prebuilt = '/home/jasper/Desktop/different_versions_spectrometer/8bram_complex_spec_fft_test/fft_test_complex_spec/outputs/fft_test_complex_spec_2024-09-12_0926.fpg'
        else:
            fpg_prebuilt = '/home/jasper/Desktop/different_versions_spectrometer/8bram_complex_spec_fft_test/fft_test_complex_spec/outputs/fft_test_complex_spec_2024-09-12_0926.fpg'

        print('using prebuilt fpg file at %s' % fpg_prebuilt)
        bitstream = fpg_prebuilt

    print('Connecting to %s... ' % (hostname))
    fpga = casperfpga.CasperFpga(hostname)
    time.sleep(0.2)

    if not opts.skip:
        print('Programming FPGA with %s...' % bitstream)
        fpga.upload_to_ram_and_program(bitstream)
        print('done')
    else:
        fpga.get_system_information()
        print('skip programming fpga...')

    fpga_rfdc = fpga.adcs['rfdc']
    fpga_rfdc.init()
    time.sleep(1)
    fpga_rfdc.status()

    c = fpga_rfdc.show_clk_files()
    fpga_rfdc.progpll('lmk', c[1])
    fpga_rfdc.progpll('lmx', c[0])
    print('Configuring accumulation period...')
    fpga.write_int('acc_len', acc_len)
    time.sleep(0.1)
    print('Done')

    print('Setting attenuation parameter...')
    fpga.write_int('atten_mult_re', ((2**32) - 1))
    time.sleep(1)
    print('Done')

    print('Resetting counters...')
    fpga.write_int('cnt_rst', 1)
    fpga.write_int('cnt_rst', 0)
    time.sleep(5)
    print('Done')

    bram_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']
    try:
        plot_spectrum(fpga, bram_names, cx=mode)
    except KeyboardInterrupt:
        exit()

