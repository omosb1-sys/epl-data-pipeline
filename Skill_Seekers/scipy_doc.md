# Scipy - Doc

**Pages:** 200

---

## Random Number Generators (scipy.stats.sampling)#

**URL:** https://docs.scipy.org/doc/scipy/reference/stats.sampling.html

**Contents:**
- Random Number Generators (scipy.stats.sampling)#
- Generators Wrapped#
  - For continuous distributions#
  - For discrete distributions#
  - Warnings / Errors used in scipy.stats.sampling#
- Generators for pre-defined distributions#

This module contains a collection of random number generators to sample from univariate continuous and discrete distributions. It uses the implementation of a C library called “UNU.RAN”. The only exception is RatioUniforms, which is a pure Python implementation of the Ratio-of-Uniforms method.

NumericalInverseHermite(dist, *[, domain, ...])

Hermite interpolation based INVersion of CDF (HINV).

NumericalInversePolynomial(dist, *[, mode, ...])

Polynomial interpolation based INVersion of CDF (PINV).

TransformedDensityRejection(dist, *[, mode, ...])

Transformed Density Rejection (TDR) Method.

SimpleRatioUniforms(dist, *[, mode, ...])

Simple Ratio-of-Uniforms (SROU) Method.

RatioUniforms(pdf, *, umax, vmin, vmax[, c, ...])

Generate random samples from a probability density function using the ratio-of-uniforms method.

DiscreteAliasUrn(dist, *[, domain, ...])

Discrete Alias-Urn Method.

DiscreteGuideTable(dist, *[, domain, ...])

Discrete Guide Table method.

Raised when an error occurs in the UNU.RAN library.

To easily apply the above methods for some of the continuous distributions in scipy.stats, the following functionality can be used:

FastGeneratorInversion(dist, *[, domain, ...])

Fast sampling by numerical inversion of the CDF for a large class of continuous distributions in scipy.stats.

---

## Signal processing (scipy.signal)#

**URL:** https://docs.scipy.org/doc/scipy/reference/signal.html

**Contents:**
- Signal processing (scipy.signal)#
- Convolution#
- B-splines#
- Filtering#
- Filter design#
- Matlab-style IIR filter design#
- Continuous-time linear systems#
- Discrete-time linear systems#
- LTI representations#
- Waveforms#

convolve(in1, in2[, mode, method])

Convolve two N-dimensional arrays.

correlate(in1, in2[, mode, method])

Cross-correlate two N-dimensional arrays.

fftconvolve(in1, in2[, mode, axes])

Convolve two N-dimensional arrays using FFT.

oaconvolve(in1, in2[, mode, axes])

Convolve two N-dimensional arrays using the overlap-add method.

convolve2d(in1, in2[, mode, boundary, fillvalue])

Convolve two 2-dimensional arrays.

correlate2d(in1, in2[, mode, boundary, ...])

Cross-correlate two 2-dimensional arrays.

sepfir2d(input, hrow, hcol)

Convolve with a 2-D separable FIR filter.

choose_conv_method(in1, in2[, mode, measure])

Find the fastest convolution/correlation method.

correlation_lags(in1_len, in2_len[, mode])

Calculates the lag / displacement indices array for 1D cross-correlation.

Gaussian approximation to B-spline basis function of order n.

cspline1d(signal[, lamb])

Compute cubic spline coefficients for rank-1 array.

qspline1d(signal[, lamb])

Compute quadratic spline coefficients for rank-1 array.

cspline2d(signal[, lamb, precision])

Coefficients for 2-D cubic (3rd order) B-spline.

qspline2d(signal[, lamb, precision])

Coefficients for 2-D quadratic (2nd order) B-spline.

cspline1d_eval(cj, newx[, dx, x0])

Evaluate a cubic spline at the new set of points.

qspline1d_eval(cj, newx[, dx, x0])

Evaluate a quadratic spline at the new set of points.

spline_filter(Iin[, lmbda])

Smoothing spline (cubic) filtering of a rank-2 array.

order_filter(a, domain, rank)

Perform an order filter on an N-D array.

medfilt(volume[, kernel_size])

Perform a median filter on an N-dimensional array.

medfilt2d(input[, kernel_size])

Median filter a 2-dimensional array.

wiener(im[, mysize, noise])

Perform a Wiener filter on an N-dimensional array.

symiirorder1(signal, c0, z1[, precision])

Implement a smoothing IIR filter with mirror-symmetric boundary conditions using a cascade of first-order sections.

symiirorder2(input, r, omega[, precision])

Implement a smoothing IIR filter with mirror-symmetric boundary conditions using a cascade of second-order sections.

lfilter(b, a, x[, axis, zi])

Filter data along one-dimension with an IIR or FIR filter.

lfiltic(b, a, y[, x])

Construct initial conditions for lfilter given input and output vectors.

Construct initial conditions for lfilter for step response steady-state.

filtfilt(b, a, x[, axis, padtype, padlen, ...])

Apply a digital filter forward and backward to a signal.

savgol_filter(x, window_length, polyorder[, ...])

Apply a Savitzky-Golay filter to an array.

deconvolve(signal, divisor)

Deconvolves divisor out of signal using inverse filtering.

sosfilt(sos, x[, axis, zi])

Filter data along one dimension using cascaded second-order sections.

Construct initial conditions for sosfilt for step response steady-state.

sosfiltfilt(sos, x[, axis, padtype, padlen])

A forward-backward digital filter using cascaded second-order sections.

hilbert(x[, N, axis])

FFT-based computation of the analytic signal.

Compute the '2-D' analytic signal of x

envelope(z[, bp_in, n_out, squared, ...])

Compute the envelope of a real- or complex-valued signal.

decimate(x, q[, n, ftype, axis, zero_phase])

Downsample the signal after applying an anti-aliasing filter.

detrend(data[, axis, type, bp, overwrite_data])

Remove linear or constant trend along axis from data.

resample(x, num[, t, axis, window, domain])

Resample x to num samples using the Fourier method along the given axis.

resample_poly(x, up, down[, axis, window, ...])

Resample x along the given axis using polyphase filtering.

upfirdn(h, x[, up, down, axis, mode, cval])

Upsample, FIR filter, and downsample.

Calculate a digital IIR filter from an analog transfer function by utilizing the bilinear transform.

bilinear_zpk(z, p, k, fs)

Return a digital IIR filter from an analog one using a bilinear transform.

findfreqs(num, den, N[, kind])

Find array of frequencies for computing the response of an analog filter.

firls(numtaps, bands, desired, *[, weight, fs])

FIR filter design using least-squares error minimization.

firwin(numtaps, cutoff, *[, width, window, ...])

FIR filter design using the window method.

firwin2(numtaps, freq, gain, *[, nfreqs, ...])

FIR filter design using the window method.

firwin_2d(hsize, window, *[, fc, fs, ...])

2D FIR filter design using the window method.

freqs(b, a[, worN, plot])

Compute frequency response of analog filter.

freqs_zpk(z, p, k[, worN])

Compute frequency response of analog filter.

freqz(b[, a, worN, whole, plot, fs, ...])

Compute the frequency response of a digital filter.

sosfreqz(*args, **kwargs)

Compute the frequency response of a digital filter in SOS format (legacy).

freqz_sos(sos[, worN, whole, fs])

Compute the frequency response of a digital filter in SOS format.

freqz_zpk(z, p, k[, worN, whole, fs])

Compute the frequency response of a digital filter in ZPK form.

gammatone(freq, ftype[, order, numtaps, fs])

Gammatone filter design.

group_delay(system[, w, whole, fs])

Compute the group delay of a digital filter.

iirdesign(wp, ws, gpass, gstop[, analog, ...])

Complete IIR digital and analog filter design.

iirfilter(N, Wn[, rp, rs, btype, analog, ...])

IIR digital and analog filter design given order and critical points.

kaiser_atten(numtaps, width)

Compute the attenuation of a Kaiser FIR filter.

Compute the Kaiser parameter beta, given the attenuation a.

kaiserord(ripple, width)

Determine the filter window parameters for the Kaiser window method.

minimum_phase(h[, method, n_fft, half])

Convert a linear-phase FIR filter to minimum phase

savgol_coeffs(window_length, polyorder[, ...])

Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

remez(numtaps, bands, desired, *[, weight, ...])

Calculate the minimax optimal filter using the Remez exchange algorithm.

unique_roots(p[, tol, rtype])

Determine unique roots and their multiplicities from a list of roots.

residue(b, a[, tol, rtype])

Compute partial-fraction expansion of b(s) / a(s).

residuez(b, a[, tol, rtype])

Compute partial-fraction expansion of b(z) / a(z).

invres(r, p, k[, tol, rtype])

Compute b(s) and a(s) from partial fraction expansion.

invresz(r, p, k[, tol, rtype])

Compute b(z) and a(z) from partial fraction expansion.

Warning about badly conditioned filter coefficients

Lower-level filter design functions:

abcd_normalize([A, B, C, D])

Check state-space matrices and ensure they are 2-D.

band_stop_obj(wp, ind, passb, stopb, gpass, ...)

Band Stop Objective Function for order minimization.

Return (z,p,k) for analog prototype of an Nth-order Bessel filter.

Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

Return (z,p,k) for Nth-order Chebyshev type II analog lowpass filter.

Return (z,p,k) of Nth-order elliptic analog lowpass filter.

lp2bp(b, a[, wo, bw])

Transform a lowpass filter prototype to a bandpass filter.

lp2bp_zpk(z, p, k[, wo, bw])

Transform a lowpass filter prototype to a bandpass filter.

lp2bs(b, a[, wo, bw])

Transform a lowpass filter prototype to a bandstop filter.

lp2bs_zpk(z, p, k[, wo, bw])

Transform a lowpass filter prototype to a bandstop filter.

Transform a lowpass filter prototype to a highpass filter.

lp2hp_zpk(z, p, k[, wo])

Transform a lowpass filter prototype to a highpass filter.

Transform a lowpass filter prototype to a different frequency.

lp2lp_zpk(z, p, k[, wo])

Transform a lowpass filter prototype to a different frequency.

Normalize numerator/denominator of a continuous-time transfer function.

butter(N, Wn[, btype, analog, output, fs])

Butterworth digital and analog filter design.

buttord(wp, ws, gpass, gstop[, analog, fs])

Butterworth filter order selection.

cheby1(N, rp, Wn[, btype, analog, output, fs])

Chebyshev type I digital and analog filter design.

cheb1ord(wp, ws, gpass, gstop[, analog, fs])

Chebyshev type I filter order selection.

cheby2(N, rs, Wn[, btype, analog, output, fs])

Chebyshev type II digital and analog filter design.

cheb2ord(wp, ws, gpass, gstop[, analog, fs])

Chebyshev type II filter order selection.

ellip(N, rp, rs, Wn[, btype, analog, output, fs])

Elliptic (Cauer) digital and analog filter design.

ellipord(wp, ws, gpass, gstop[, analog, fs])

Elliptic (Cauer) filter order selection.

bessel(N, Wn[, btype, analog, output, norm, fs])

Bessel/Thomson digital and analog filter design.

iirnotch(w0, Q[, fs])

Design second-order IIR notch digital filter.

Design second-order IIR peak (resonant) digital filter.

iircomb(w0, Q[, ftype, fs, pass_zero])

Design IIR notching or peaking digital comb filter.

Continuous-time linear time invariant system base class.

StateSpace(*system, **kwargs)

Linear Time Invariant system in state-space form.

TransferFunction(*system, **kwargs)

Linear Time Invariant system class in transfer function form.

ZerosPolesGain(*system, **kwargs)

Linear Time Invariant system class in zeros, poles, gain form.

lsim(system, U, T[, X0, interp])

Simulate output of a continuous-time linear system.

impulse(system[, X0, T, N])

Impulse response of continuous-time system.

step(system[, X0, T, N])

Step response of continuous-time system.

freqresp(system[, w, n])

Calculate the frequency response of a continuous-time system.

Calculate Bode magnitude and phase data of a continuous-time system.

dlti(*system, **kwargs)

Discrete-time linear time invariant system base class.

StateSpace(*system, **kwargs)

Linear Time Invariant system in state-space form.

TransferFunction(*system, **kwargs)

Linear Time Invariant system class in transfer function form.

ZerosPolesGain(*system, **kwargs)

Linear Time Invariant system class in zeros, poles, gain form.

dlsim(system, u[, t, x0])

Simulate output of a discrete-time linear system.

dimpulse(system[, x0, t, n])

Impulse response of discrete-time system.

dstep(system[, x0, t, n])

Step response of discrete-time system.

dfreqresp(system[, w, n, whole])

Calculate the frequency response of a discrete-time system.

dbode(system[, w, n])

Calculate Bode magnitude and phase data of a discrete-time system.

Return zero, pole, gain (z, p, k) representation from a numerator, denominator representation of a linear filter.

tf2sos(b, a[, pairing, analog])

Return second-order sections from transfer function representation

Transfer function to state-space representation.

Return polynomial transfer function representation from zeros and poles

zpk2sos(z, p, k[, pairing, analog])

Return second-order sections from zeros, poles, and gain of a system

Zero-pole-gain representation to state-space representation

ss2tf(A, B, C, D[, input])

State-space to transfer function.

ss2zpk(A, B, C, D[, input])

State-space representation to zero-pole-gain representation.

Return zeros, poles, and gain of a series of second-order sections

Return a single transfer function from a series of second-order sections

cont2discrete(system, dt[, method, alpha])

Transform a continuous to a discrete state-space system.

place_poles(A, B, poles[, method, rtol, maxiter])

Compute K such that eigenvalues (A - dot(B, K))=poles.

chirp(t, f0, t1, f1[, method, phi, ...])

Frequency-swept cosine generator.

gausspulse(t[, fc, bw, bwr, tpr, retquad, ...])

Return a Gaussian modulated sinusoid:

max_len_seq(nbits[, state, length, taps])

Maximum length sequence (MLS) generator.

Return a periodic sawtooth or triangle waveform.

Return a periodic square-wave waveform.

sweep_poly(t, poly[, phi])

Frequency-swept cosine generator, with a time-dependent frequency.

unit_impulse(shape[, idx, dtype])

Unit impulse signal (discrete delta function) or unit basis vector.

For window functions, see the scipy.signal.windows namespace.

In the scipy.signal namespace, there is a convenience function to obtain these windows by name:

get_window(window, Nx[, fftbins, xp, device])

Return a window of a given length and type.

argrelmin(data[, axis, order, mode])

Calculate the relative minima of data.

argrelmax(data[, axis, order, mode])

Calculate the relative maxima of data.

argrelextrema(data, comparator[, axis, ...])

Calculate the relative extrema of data.

find_peaks(x[, height, threshold, distance, ...])

Find peaks inside a signal based on peak properties.

find_peaks_cwt(vector, widths[, wavelet, ...])

Find peaks in a 1-D array with wavelet transformation.

peak_prominences(x, peaks[, wlen])

Calculate the prominence of each peak in a signal.

peak_widths(x, peaks[, rel_height, ...])

Calculate the width of each peak in a signal.

periodogram(x[, fs, window, nfft, detrend, ...])

Estimate power spectral density using a periodogram.

welch(x[, fs, window, nperseg, noverlap, ...])

Estimate power spectral density using Welch's method.

csd(x, y[, fs, window, nperseg, noverlap, ...])

Estimate the cross power spectral density, Pxy, using Welch's method.

coherence(x, y[, fs, window, nperseg, ...])

Estimate the magnitude squared coherence estimate, Cxy, of discrete-time signals X and Y using Welch's method.

spectrogram(x[, fs, window, nperseg, ...])

Compute a spectrogram with consecutive Fourier transforms (legacy function).

lombscargle(x, y, freqs[, precenter, ...])

Compute the generalized Lomb-Scargle periodogram.

vectorstrength(events, period)

Determine the vector strength of the events corresponding to the given period.

ShortTimeFFT(win, hop, fs, *[, fft_mode, ...])

Provide a parametrized discrete Short-time Fourier transform (stft) and its inverse (istft).

closest_STFT_dual_window(win, hop[, ...])

Calculate the STFT dual window of a given window closest to a desired dual window.

stft(x[, fs, window, nperseg, noverlap, ...])

Compute the Short Time Fourier Transform (legacy function).

istft(Zxx[, fs, window, nperseg, noverlap, ...])

Perform the inverse Short Time Fourier transform (legacy function).

check_COLA(window, nperseg, noverlap[, tol])

Check whether the Constant OverLap Add (COLA) constraint is met (legacy function).

check_NOLA(window, nperseg, noverlap[, tol])

Check whether the Nonzero Overlap Add (NOLA) constraint is met.

czt(x[, m, w, a, axis])

Compute the frequency response around a spiral in the Z plane.

zoom_fft(x, fn[, m, fs, endpoint, axis])

Compute the DFT of x only for frequencies in range fn.

Create a callable chirp z-transform function.

ZoomFFT(n, fn[, m, fs, endpoint])

Create a callable zoom FFT transform function.

czt_points(m[, w, a])

Return the points at which the chirp z-transform is computed.

The functions are simpler to use than the classes, but are less efficient when using the same transform on many arrays of the same length, since they repeatedly generate the same chirp signal with every call. In these cases, use the classes to create a reusable function instead.

---

## idst#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idst.html

**Contents:**
- idst#

Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

Type of the DST (see Notes). Default type is 2.

Length of the transform. If n < x.shape[axis], x is truncated. If n > x.shape[axis], x is zero-padded. The default results in n = x.shape[axis].

Axis along which the idst is computed; the default is over the last axis (i.e., axis=-1).

Normalization mode (see Notes). Default is “backward”.

If True, the contents of x can be destroyed; the default is False.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

Whether to use the orthogonalized IDST variant (see Notes). Defaults to True when norm="ortho" and False otherwise.

Added in version 1.8.0.

The transformed input array.

For type in {2, 3}, norm="ortho" breaks the direct correspondence with the inverse direct Fourier transform.

For norm="ortho" both the dst and idst are scaled by the same overall factor in both directions. By default, the transform is also orthogonalized which for types 2 and 3 means the transform definition is modified to give orthogonality of the DST matrix (see dst for the full definitions).

‘The’ IDST is the IDST-II, which is the same as the normalized DST-III.

The IDST is equivalent to a normal DST except for the normalization and type. DST type 1 and 4 are their own inverse and DSTs 2 and 3 are each other’s inverses.

---

## Multidimensional Image Processing (scipy.ndimage)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/ndimage.html

**Contents:**
- Multidimensional Image Processing (scipy.ndimage)#
- Introduction#
- Properties shared by all functions#
- Filter functions#
  - Correlation and convolution#
  - Smoothing filters#
  - Filters based on order statistics#
  - Derivatives#
  - Generic filter functions#
  - Fourier domain filters#

Image processing and analysis are generally seen as operations on 2-D arrays of values. There are, however, a number of fields where images of higher dimensionality must be analyzed. Good examples of these are medical imaging and biological imaging. numpy is suited very well for this type of applications due to its inherent multidimensional nature. The scipy.ndimage packages provides a number of general image processing and analysis functions that are designed to operate with arrays of arbitrary dimensionality. The packages currently includes: functions for linear and non-linear filtering, binary morphology, B-spline interpolation, and object measurements.

All functions share some common properties. Notably, all functions allow the specification of an output array with the output argument. With this argument, you can specify an array that will be changed in-place with the result with the operation. In this case, the result is not returned. Usually, using the output argument is more efficient, since an existing array is used to store the result.

The type of arrays returned is dependent on the type of operation, but it is, in most cases, equal to the type of the input. If, however, the output argument is used, the type of the result is equal to the type of the specified output argument. If no output argument is given, it is still possible to specify what the result of the output should be. This is done by simply assigning the desired numpy type object to the output argument. For example:

The functions described in this section all perform some type of spatial filtering of the input array: the elements in the output are some function of the values in the neighborhood of the corresponding input element. We refer to this neighborhood of elements as the filter kernel, which is often rectangular in shape but may also have an arbitrary footprint. Many of the functions described below allow you to define the footprint of the kernel by passing a mask through the footprint parameter. For example, a cross-shaped kernel can be defined as follows:

Usually, the origin of the kernel is at the center calculated by dividing the dimensions of the kernel shape by two. For instance, the origin of a 1-D kernel of length three is at the second element. Take, for example, the correlation of a 1-D array with a filter of length 3 consisting of ones:

Sometimes, it is convenient to choose a different origin for the kernel. For this reason, most functions support the origin parameter, which gives the origin of the filter relative to its center. For example:

The effect is a shift of the result towards the left. This feature will not be needed very often, but it may be useful, especially for filters that have an even size. A good example is the calculation of backward and forward differences:

We could also have calculated the forward difference as follows:

However, using the origin parameter instead of a larger kernel is more efficient. For multidimensional kernels, origin can be a number, in which case the origin is assumed to be equal along all axes, or a sequence giving the origin along each axis.

Since the output elements are a function of elements in the neighborhood of the input elements, the borders of the array need to be dealt with appropriately by providing the values outside the borders. This is done by assuming that the arrays are extended beyond their boundaries according to certain boundary conditions. In the functions described below, the boundary conditions can be selected using the mode parameter, which must be a string with the name of the boundary condition. The following boundary conditions are currently supported:

use the value at the boundary

periodically replicate the array

reflect the array at the boundary

mirror the array at the boundary

use a constant value, default is 0.0

The following synonyms are also supported for consistency with the interpolation routines:

equivalent to “constant”*

equivalent to “reflect”

* “grid-constant” and “constant” are equivalent for filtering operations, but have different behavior in interpolation functions. For API consistency, the filtering functions accept either name.

The “constant” mode is special since it needs an additional parameter to specify the constant value that should be used.

Note that modes mirror and reflect differ only in whether the sample at the boundary is repeated upon reflection. For mode mirror, the point of symmetry is exactly at the final sample, so that value is not repeated. This mode is also known as whole-sample symmetric since the point of symmetry falls on the final sample. Similarly, reflect is often referred to as half-sample symmetric as the point of symmetry is half a sample beyond the array boundary.

The easiest way to implement such boundary conditions would be to copy the data to a larger array and extend the data at the borders according to the boundary conditions. For large arrays and large filter kernels, this would be very memory consuming, and the functions described below, therefore, use a different approach that does not require allocating large temporary buffers.

The correlate1d function calculates a 1-D correlation along the given axis. The lines of the array along the given axis are correlated with the given weights. The weights parameter must be a 1-D sequence of numbers.

The function correlate implements multidimensional correlation of the input array with a given kernel.

The convolve1d function calculates a 1-D convolution along the given axis. The lines of the array along the given axis are convoluted with the given weights. The weights parameter must be a 1-D sequence of numbers.

The function convolve implements multidimensional convolution of the input array with a given kernel.

A convolution is essentially a correlation after mirroring the kernel. As a result, the origin parameter behaves differently than in the case of a correlation: the results is shifted in the opposite direction.

The gaussian_filter1d function implements a 1-D Gaussian filter. The standard deviation of the Gaussian filter is passed through the parameter sigma. Setting order = 0 corresponds to convolution with a Gaussian kernel. An order of 1, 2, or 3 corresponds to convolution with the first, second, or third derivatives of a Gaussian. Higher-order derivatives are not implemented.

The gaussian_filter function implements a multidimensional Gaussian filter. The standard deviations of the Gaussian filter along each axis are passed through the parameter sigma as a sequence or numbers. If sigma is not a sequence but a single number, the standard deviation of the filter is equal along all directions. The order of the filter can be specified separately for each axis. An order of 0 corresponds to convolution with a Gaussian kernel. An order of 1, 2, or 3 corresponds to convolution with the first, second, or third derivatives of a Gaussian. Higher-order derivatives are not implemented. The order parameter must be a number, to specify the same order for all axes, or a sequence of numbers to specify a different order for each axis. The example below shows the filter applied on test data with different values of sigma. The order parameter is kept at 0.

The multidimensional filter is implemented as a sequence of 1-D Gaussian filters. The intermediate arrays are stored in the same data type as the output. Therefore, for output types with a lower precision, the results may be imprecise because intermediate results may be stored with insufficient precision. This can be prevented by specifying a more precise output type.

The uniform_filter1d function calculates a 1-D uniform filter of the given size along the given axis.

The uniform_filter implements a multidimensional uniform filter. The sizes of the uniform filter are given for each axis as a sequence of integers by the size parameter. If size is not a sequence, but a single number, the sizes along all axes are assumed to be equal.

The multidimensional filter is implemented as a sequence of 1-D uniform filters. The intermediate arrays are stored in the same data type as the output. Therefore, for output types with a lower precision, the results may be imprecise because intermediate results may be stored with insufficient precision. This can be prevented by specifying a more precise output type.

The minimum_filter1d function calculates a 1-D minimum filter of the given size along the given axis.

The maximum_filter1d function calculates a 1-D maximum filter of the given size along the given axis.

The minimum_filter function calculates a multidimensional minimum filter. Either the sizes of a rectangular kernel or the footprint of the kernel must be provided. The size parameter, if provided, must be a sequence of sizes or a single number, in which case the size of the filter is assumed to be equal along each axis. The footprint, if provided, must be an array that defines the shape of the kernel by its non-zero elements.

The maximum_filter function calculates a multidimensional maximum filter. Either the sizes of a rectangular kernel or the footprint of the kernel must be provided. The size parameter, if provided, must be a sequence of sizes or a single number, in which case the size of the filter is assumed to be equal along each axis. The footprint, if provided, must be an array that defines the shape of the kernel by its non-zero elements.

The rank_filter function calculates a multidimensional rank filter. The rank may be less than zero, i.e., rank = -1 indicates the largest element. Either the sizes of a rectangular kernel or the footprint of the kernel must be provided. The size parameter, if provided, must be a sequence of sizes or a single number, in which case the size of the filter is assumed to be equal along each axis. The footprint, if provided, must be an array that defines the shape of the kernel by its non-zero elements.

The percentile_filter function calculates a multidimensional percentile filter. The percentile may be less than zero, i.e., percentile = -20 equals percentile = 80. Either the sizes of a rectangular kernel or the footprint of the kernel must be provided. The size parameter, if provided, must be a sequence of sizes or a single number, in which case the size of the filter is assumed to be equal along each axis. The footprint, if provided, must be an array that defines the shape of the kernel by its non-zero elements.

The median_filter function calculates a multidimensional median filter. Either the sizes of a rectangular kernel or the footprint of the kernel must be provided. The size parameter, if provided, must be a sequence of sizes or a single number, in which case the size of the filter is assumed to be equal along each axis. The footprint if provided, must be an array that defines the shape of the kernel by its non-zero elements.

Derivative filters can be constructed in several ways. The function gaussian_filter1d, described in Smoothing filters, can be used to calculate derivatives along a given axis using the order parameter. Other derivative filters are the Prewitt and Sobel filters:

The prewitt function calculates a derivative along the given axis.

The sobel function calculates a derivative along the given axis.

The Laplace filter is calculated by the sum of the second derivatives along all axes. Thus, different Laplace filters can be constructed using different second-derivative functions. Therefore, we provide a general function that takes a function argument to calculate the second derivative along a given direction.

The function generic_laplace calculates a Laplace filter using the function passed through derivative2 to calculate second derivatives. The function derivative2 should have the following signature

It should calculate the second derivative along the dimension axis. If output is not None, it should use that for the output and return None, otherwise it should return the result. mode, cval have the usual meaning.

The extra_arguments and extra_keywords arguments can be used to pass a tuple of extra arguments and a dictionary of named arguments that are passed to derivative2 at each call.

To demonstrate the use of the extra_arguments argument, we could do

The following two functions are implemented using generic_laplace by providing appropriate functions for the second-derivative function:

The function laplace calculates the Laplace using discrete differentiation for the second derivative (i.e., convolution with [1, -2, 1]).

The function gaussian_laplace calculates the Laplace filter using gaussian_filter to calculate the second derivatives. The standard deviations of the Gaussian filter along each axis are passed through the parameter sigma as a sequence or numbers. If sigma is not a sequence but a single number, the standard deviation of the filter is equal along all directions.

The gradient magnitude is defined as the square root of the sum of the squares of the gradients in all directions. Similar to the generic Laplace function, there is a generic_gradient_magnitude function that calculates the gradient magnitude of an array.

The function generic_gradient_magnitude calculates a gradient magnitude using the function passed through derivative to calculate first derivatives. The function derivative should have the following signature

It should calculate the derivative along the dimension axis. If output is not None, it should use that for the output and return None, otherwise it should return the result. mode, cval have the usual meaning.

The extra_arguments and extra_keywords arguments can be used to pass a tuple of extra arguments and a dictionary of named arguments that are passed to derivative at each call.

For example, the sobel function fits the required signature

See the documentation of generic_laplace for examples of using the extra_arguments and extra_keywords arguments.

The sobel and prewitt functions fit the required signature and can, therefore, be used directly with generic_gradient_magnitude.

The function gaussian_gradient_magnitude calculates the gradient magnitude using gaussian_filter to calculate the first derivatives. The standard deviations of the Gaussian filter along each axis are passed through the parameter sigma as a sequence or numbers. If sigma is not a sequence but a single number, the standard deviation of the filter is equal along all directions.

To implement filter functions, generic functions can be used that accept a callable object that implements the filtering operation. The iteration over the input and output arrays is handled by these generic functions, along with such details as the implementation of the boundary conditions. Only a callable object implementing a callback function that does the actual filtering work must be provided. The callback function can also be written in C and passed using a PyCapsule (see Extending scipy.ndimage in C for more information).

The generic_filter1d function implements a generic 1-D filter function, where the actual filtering operation must be supplied as a python function (or other callable object). The generic_filter1d function iterates over the lines of an array and calls function at each line. The arguments that are passed to function are 1-D arrays of the numpy.float64 type. The first contains the values of the current line. It is extended at the beginning and the end, according to the filter_size and origin arguments. The second array should be modified in-place to provide the output values of the line. For example, consider a correlation along one dimension:

The same operation can be implemented using generic_filter1d, as follows:

Here, the origin of the kernel was (by default) assumed to be in the middle of the filter of length 3. Therefore, each input line had been extended by one value at the beginning and at the end, before the function was called.

Optionally, extra arguments can be defined and passed to the filter function. The extra_arguments and extra_keywords arguments can be used to pass a tuple of extra arguments and/or a dictionary of named arguments that are passed to derivative at each call. For example, we can pass the parameters of our filter as an argument

The generic_filter function implements a generic filter function, where the actual filtering operation must be supplied as a python function (or other callable object). The generic_filter function iterates over the array and calls function at each element. The argument of function is a 1-D array of the numpy.float64 type that contains the values around the current element that are within the footprint of the filter. The function should return a single value that can be converted to a double precision number. For example, consider a correlation:

The same operation can be implemented using generic_filter, as follows:

Here, a kernel footprint was specified that contains only two elements. Therefore, the filter function receives a buffer of length equal to two, which was multiplied with the proper weights and the result summed.

When calling generic_filter, either the sizes of a rectangular kernel or the footprint of the kernel must be provided. The size parameter, if provided, must be a sequence of sizes or a single number, in which case the size of the filter is assumed to be equal along each axis. The footprint, if provided, must be an array that defines the shape of the kernel by its non-zero elements.

Optionally, extra arguments can be defined and passed to the filter function. The extra_arguments and extra_keywords arguments can be used to pass a tuple of extra arguments and/or a dictionary of named arguments that are passed to derivative at each call. For example, we can pass the parameters of our filter as an argument

These functions iterate over the lines or elements starting at the last axis, i.e., the last index changes the fastest. This order of iteration is guaranteed for the case that it is important to adapt the filter depending on spatial location. Here is an example of using a class that implements the filter and keeps track of the current coordinates while iterating. It performs the same filter operation as described above for generic_filter, but additionally prints the current coordinates:

For the generic_filter1d function, the same approach works, except that this function does not iterate over the axis that is being filtered. The example for generic_filter1d then becomes this:

The functions described in this section perform filtering operations in the Fourier domain. Thus, the input array of such a function should be compatible with an inverse Fourier transform function, such as the functions from the numpy.fft module. We, therefore, have to deal with arrays that may be the result of a real or a complex Fourier transform. In the case of a real Fourier transform, only half of the of the symmetric complex transform is stored. Additionally, it needs to be known what the length of the axis was that was transformed by the real fft. The functions described here provide a parameter n that, in the case of a real transform, must be equal to the length of the real transform axis before transformation. If this parameter is less than zero, it is assumed that the input array was the result of a complex Fourier transform. The parameter axis can be used to indicate along which axis the real transform was executed.

The fourier_shift function multiplies the input array with the multidimensional Fourier transform of a shift operation for the given shift. The shift parameter is a sequence of shifts for each dimension or a single value for all dimensions.

The fourier_gaussian function multiplies the input array with the multidimensional Fourier transform of a Gaussian filter with given standard deviations sigma. The sigma parameter is a sequence of values for each dimension or a single value for all dimensions.

The fourier_uniform function multiplies the input array with the multidimensional Fourier transform of a uniform filter with given sizes size. The size parameter is a sequence of values for each dimension or a single value for all dimensions.

The fourier_ellipsoid function multiplies the input array with the multidimensional Fourier transform of an elliptically-shaped filter with given sizes size. The size parameter is a sequence of values for each dimension or a single value for all dimensions. This function is only implemented for dimensions 1, 2, and 3.

This section describes various interpolation functions that are based on B-spline theory. A good introduction to B-splines can be found in [1] with detailed algorithms for image interpolation given in [5].

Interpolation using splines of an order larger than 1 requires a pre-filtering step. The interpolation functions described in section Interpolation functions apply pre-filtering by calling spline_filter, but they can be instructed not to do this by setting the prefilter keyword equal to False. This is useful if more than one interpolation operation is done on the same array. In this case, it is more efficient to do the pre-filtering only once and use a pre-filtered array as the input of the interpolation functions. The following two functions implement the pre-filtering:

The spline_filter1d function calculates a 1-D spline filter along the given axis. An output array can optionally be provided. The order of the spline must be larger than 1 and less than 6.

The spline_filter function calculates a multidimensional spline filter.

The multidimensional filter is implemented as a sequence of 1-D spline filters. The intermediate arrays are stored in the same data type as the output. Therefore, if an output with a limited precision is requested, the results may be imprecise because intermediate results may be stored with insufficient precision. This can be prevented by specifying a output type of high precision.

The interpolation functions all employ spline interpolation to effect some type of geometric transformation of the input array. This requires a mapping of the output coordinates to the input coordinates, and therefore, the possibility arises that input values outside the boundaries may be needed. This problem is solved in the same way as described in Filter functions for the multidimensional filter functions. Therefore, these functions all support a mode parameter that determines how the boundaries are handled, and a cval parameter that gives a constant value in case that the ‘constant’ mode is used. The behavior of all modes, including at non-integer locations is illustrated below. Note the boundaries are not handled the same for all modes; reflect (aka grid-mirror) and grid-wrap involve symmetry or repetition about a point that is half way between image samples (dashed vertical lines) while modes mirror and wrap treat the image as if it’s extent ends exactly at the first and last sample point rather than 0.5 samples past it.

The coordinates of image samples fall on integer sampling locations in the range from 0 to shape[i] - 1 along each axis, i. The figure below illustrates the interpolation of a point at location (3.7, 3.3) within an image of shape (7, 7). For an interpolation of order n, n + 1 samples are involved along each axis. The filled circles illustrate the sampling locations involved in the interpolation of the value at the location of the red x.

The geometric_transform function applies an arbitrary geometric transform to the input. The given mapping function is called at each point in the output to find the corresponding coordinates in the input. mapping must be a callable object that accepts a tuple of length equal to the output array rank and returns the corresponding input coordinates as a tuple of length equal to the input array rank. The output shape and output type can optionally be provided. If not given, they are equal to the input shape and type.

Optionally, extra arguments can be defined and passed to the filter function. The extra_arguments and extra_keywords arguments can be used to pass a tuple of extra arguments and/or a dictionary of named arguments that are passed to derivative at each call. For example, we can pass the shifts in our example as arguments

The mapping function can also be written in C and passed using a scipy.LowLevelCallable. See Extending scipy.ndimage in C for more information.

The function map_coordinates applies an arbitrary coordinate transformation using the given array of coordinates. The shape of the output is derived from that of the coordinate array by dropping the first axis. The parameter coordinates is used to find for each point in the output the corresponding coordinates in the input. The values of coordinates along the first axis are the coordinates in the input array at which the output value is found. (See also the numarray coordinates function.) Since the coordinates may be non- integer coordinates, the value of the input at these coordinates is determined by spline interpolation of the requested order.

Here is an example that interpolates a 2D array at (0.5, 0.5) and (1, 2):

The affine_transform function applies an affine transformation to the input array. The given transformation matrix and offset are used to find for each point in the output the corresponding coordinates in the input. The value of the input at the calculated coordinates is determined by spline interpolation of the requested order. The transformation matrix must be 2-D or can also be given as a 1-D sequence or array. In the latter case, it is assumed that the matrix is diagonal. A more efficient interpolation algorithm is then applied that exploits the separability of the problem. The output shape and output type can optionally be provided. If not given, they are equal to the input shape and type.

The shift function returns a shifted version of the input, using spline interpolation of the requested order.

The zoom function returns a rescaled version of the input, using spline interpolation of the requested order.

The rotate function returns the input array rotated in the plane defined by the two axes given by the parameter axes, using spline interpolation of the requested order. The angle must be given in degrees. If reshape is true, then the size of the output array is adapted to contain the rotated input.

The generate_binary_structure functions generates a binary structuring element for use in binary morphology operations. The rank of the structure must be provided. The size of the structure that is returned is equal to three in each direction. The value of each element is equal to one if the square of the Euclidean distance from the element to the center is less than or equal to connectivity. For instance, 2-D 4-connected and 8-connected structures are generated as follows:

This is a visual presentation of generate_binary_structure in 3D:

Most binary morphology functions can be expressed in terms of the basic operations erosion and dilation, which can be seen here:

The binary_erosion function implements binary erosion of arrays of arbitrary rank with the given structuring element. The origin parameter controls the placement of the structuring element, as described in Filter functions. If no structuring element is provided, an element with connectivity equal to one is generated using generate_binary_structure. The border_value parameter gives the value of the array outside boundaries. The erosion is repeated iterations times. If iterations is less than one, the erosion is repeated until the result does not change anymore. If a mask array is given, only those elements with a true value at the corresponding mask element are modified at each iteration.

The binary_dilation function implements binary dilation of arrays of arbitrary rank with the given structuring element. The origin parameter controls the placement of the structuring element, as described in Filter functions. If no structuring element is provided, an element with connectivity equal to one is generated using generate_binary_structure. The border_value parameter gives the value of the array outside boundaries. The dilation is repeated iterations times. If iterations is less than one, the dilation is repeated until the result does not change anymore. If a mask array is given, only those elements with a true value at the corresponding mask element are modified at each iteration.

Here is an example of using binary_dilation to find all elements that touch the border, by repeatedly dilating an empty array from the border using the data array as the mask:

The binary_erosion and binary_dilation functions both have an iterations parameter, which allows the erosion or dilation to be repeated a number of times. Repeating an erosion or a dilation with a given structure n times is equivalent to an erosion or a dilation with a structure that is n-1 times dilated with itself. A function is provided that allows the calculation of a structure that is dilated a number of times with itself:

The iterate_structure function returns a structure by dilation of the input structure iteration - 1 times with itself.

Other morphology operations can be defined in terms of erosion and dilation. The following functions provide a few of these operations for convenience:

The binary_opening function implements binary opening of arrays of arbitrary rank with the given structuring element. Binary opening is equivalent to a binary erosion followed by a binary dilation with the same structuring element. The origin parameter controls the placement of the structuring element, as described in Filter functions. If no structuring element is provided, an element with connectivity equal to one is generated using generate_binary_structure. The iterations parameter gives the number of erosions that is performed followed by the same number of dilations.

The binary_closing function implements binary closing of arrays of arbitrary rank with the given structuring element. Binary closing is equivalent to a binary dilation followed by a binary erosion with the same structuring element. The origin parameter controls the placement of the structuring element, as described in Filter functions. If no structuring element is provided, an element with connectivity equal to one is generated using generate_binary_structure. The iterations parameter gives the number of dilations that is performed followed by the same number of erosions.

The binary_fill_holes function is used to close holes in objects in a binary image, where the structure defines the connectivity of the holes. The origin parameter controls the placement of the structuring element, as described in Filter functions. If no structuring element is provided, an element with connectivity equal to one is generated using generate_binary_structure.

The binary_hit_or_miss function implements a binary hit-or-miss transform of arrays of arbitrary rank with the given structuring elements. The hit-or-miss transform is calculated by erosion of the input with the first structure, erosion of the logical not of the input with the second structure, followed by the logical and of these two erosions. The origin parameters control the placement of the structuring elements, as described in Filter functions. If origin2 equals None, it is set equal to the origin1 parameter. If the first structuring element is not provided, a structuring element with connectivity equal to one is generated using generate_binary_structure. If structure2 is not provided, it is set equal to the logical not of structure1.

Grey-scale morphology operations are the equivalents of binary morphology operations that operate on arrays with arbitrary values. Below, we describe the grey-scale equivalents of erosion, dilation, opening and closing. These operations are implemented in a similar fashion as the filters described in Filter functions, and we refer to this section for the description of filter kernels and footprints, and the handling of array borders. The grey-scale morphology operations optionally take a structure parameter that gives the values of the structuring element. If this parameter is not given, the structuring element is assumed to be flat with a value equal to zero. The shape of the structure can optionally be defined by the footprint parameter. If this parameter is not given, the structure is assumed to be rectangular, with sizes equal to the dimensions of the structure array, or by the size parameter if structure is not given. The size parameter is only used if both structure and footprint are not given, in which case the structuring element is assumed to be rectangular and flat with the dimensions given by size. The size parameter, if provided, must be a sequence of sizes or a single number in which case the size of the filter is assumed to be equal along each axis. The footprint parameter, if provided, must be an array that defines the shape of the kernel by its non-zero elements.

Similarly to binary erosion and dilation, there are operations for grey-scale erosion and dilation:

The grey_erosion function calculates a multidimensional grey-scale erosion.

The grey_dilation function calculates a multidimensional grey-scale dilation.

Grey-scale opening and closing operations can be defined similarly to their binary counterparts:

The grey_opening function implements grey-scale opening of arrays of arbitrary rank. Grey-scale opening is equivalent to a grey-scale erosion followed by a grey-scale dilation.

The grey_closing function implements grey-scale closing of arrays of arbitrary rank. Grey-scale opening is equivalent to a grey-scale dilation followed by a grey-scale erosion.

The morphological_gradient function implements a grey-scale morphological gradient of arrays of arbitrary rank. The grey-scale morphological gradient is equal to the difference of a grey-scale dilation and a grey-scale erosion.

The morphological_laplace function implements a grey-scale morphological laplace of arrays of arbitrary rank. The grey-scale morphological laplace is equal to the sum of a grey-scale dilation and a grey-scale erosion minus twice the input.

The white_tophat function implements a white top-hat filter of arrays of arbitrary rank. The white top-hat is equal to the difference of the input and a grey-scale opening.

The black_tophat function implements a black top-hat filter of arrays of arbitrary rank. The black top-hat is equal to the difference of a grey-scale closing and the input.

Distance transforms are used to calculate the minimum distance from each element of an object to the background. The following functions implement distance transforms for three different distance metrics: Euclidean, city block, and chessboard distances.

The function distance_transform_cdt uses a chamfer type algorithm to calculate the distance transform of the input, by replacing each object element (defined by values larger than zero) with the shortest distance to the background (all non-object elements). The structure determines the type of chamfering that is done. If the structure is equal to ‘cityblock’, a structure is generated using generate_binary_structure with a squared distance equal to 1. If the structure is equal to ‘chessboard’, a structure is generated using generate_binary_structure with a squared distance equal to the rank of the array. These choices correspond to the common interpretations of the city block and the chessboard distance metrics in two dimensions.

In addition to the distance transform, the feature transform can be calculated. In this case, the index of the closest background element is returned along the first axis of the result. The return_distances, and return_indices flags can be used to indicate if the distance transform, the feature transform, or both must be returned.

The distances and indices arguments can be used to give optional output arrays that must be of the correct size and type (both numpy.int32). The basics of the algorithm used to implement this function are described in [2].

The function distance_transform_edt calculates the exact Euclidean distance transform of the input, by replacing each object element (defined by values larger than zero) with the shortest Euclidean distance to the background (all non-object elements).

In addition to the distance transform, the feature transform can be calculated. In this case, the index of the closest background element is returned along the first axis of the result. The return_distances and return_indices flags can be used to indicate if the distance transform, the feature transform, or both must be returned.

Optionally, the sampling along each axis can be given by the sampling parameter, which should be a sequence of length equal to the input rank, or a single number in which the sampling is assumed to be equal along all axes.

The distances and indices arguments can be used to give optional output arrays that must be of the correct size and type (numpy.float64 and numpy.int32).The algorithm used to implement this function is described in [3].

The function distance_transform_bf uses a brute-force algorithm to calculate the distance transform of the input, by replacing each object element (defined by values larger than zero) with the shortest distance to the background (all non-object elements). The metric must be one of “euclidean”, “cityblock”, or “chessboard”.

In addition to the distance transform, the feature transform can be calculated. In this case, the index of the closest background element is returned along the first axis of the result. The return_distances and return_indices flags can be used to indicate if the distance transform, the feature transform, or both must be returned.

Optionally, the sampling along each axis can be given by the sampling parameter, which should be a sequence of length equal to the input rank, or a single number in which the sampling is assumed to be equal along all axes. This parameter is only used in the case of the Euclidean distance transform.

The distances and indices arguments can be used to give optional output arrays that must be of the correct size and type (numpy.float64 and numpy.int32).

This function uses a slow brute-force algorithm, the function distance_transform_cdt can be used to more efficiently calculate city block and chessboard distance transforms. The function distance_transform_edt can be used to more efficiently calculate the exact Euclidean distance transform.

Segmentation is the process of separating objects of interest from the background. The most simple approach is, probably, intensity thresholding, which is easily done with numpy functions:

The result is a binary image, in which the individual objects still need to be identified and labeled. The function label generates an array where each object is assigned a unique number:

The label function generates an array where the objects in the input are labeled with an integer index. It returns a tuple consisting of the array of object labels and the number of objects found, unless the output parameter is given, in which case only the number of objects is returned. The connectivity of the objects is defined by a structuring element. For instance, in 2D using a 4-connected structuring element gives:

These two objects are not connected because there is no way in which we can place the structuring element, such that it overlaps with both objects. However, an 8-connected structuring element results in only a single object:

If no structuring element is provided, one is generated by calling generate_binary_structure (see Binary morphology) using a connectivity of one (which in 2D is the 4-connected structure of the first example). The input can be of any type, any value not equal to zero is taken to be part of an object. This is useful if you need to ‘re-label’ an array of object indices, for instance, after removing unwanted objects. Just apply the label function again to the index array. For instance:

The structuring element used by label is assumed to be symmetric.

There is a large number of other approaches for segmentation, for instance, from an estimation of the borders of the objects that can be obtained by derivative filters. One such approach is watershed segmentation. The function watershed_ift generates an array where each object is assigned a unique label, from an array that localizes the object borders, generated, for instance, by a gradient magnitude filter. It uses an array containing initial markers for the objects:

The watershed_ift function applies a watershed from markers algorithm, using Image Foresting Transform, as described in [4].

The inputs of this function are the array to which the transform is applied, and an array of markers that designate the objects by a unique label, where any non-zero value is a marker. For instance:

Here, two markers were used to designate an object (marker = 2) and the background (marker = 1). The order in which these are processed is arbitrary: moving the marker for the background to the lower-right corner of the array yields a different result:

The result is that the object (marker = 2) is smaller because the second marker was processed earlier. This may not be the desired effect if the first marker was supposed to designate a background object. Therefore, watershed_ift treats markers with a negative value explicitly as background markers and processes them after the normal markers. For instance, replacing the first marker by a negative marker gives a result similar to the first example:

The connectivity of the objects is defined by a structuring element. If no structuring element is provided, one is generated by calling generate_binary_structure (see Binary morphology) using a connectivity of one (which in 2D is a 4-connected structure.) For example, using an 8-connected structure with the last example yields a different object:

The implementation of watershed_ift limits the data types of the input to numpy.uint8 and numpy.uint16.

Given an array of labeled objects, the properties of the individual objects can be measured. The find_objects function can be used to generate a list of slices that for each object, give the smallest sub-array that fully contains the object:

The find_objects function finds all objects in a labeled array and returns a list of slices that correspond to the smallest regions in the array that contains the object.

The function find_objects returns slices for all objects, unless the max_label parameter is larger then zero, in which case only the first max_label objects are returned. If an index is missing in the label array, None is return instead of a slice. For example:

The list of slices generated by find_objects is useful to find the position and dimensions of the objects in the array, but can also be used to perform measurements on the individual objects. Say, we want to find the sum of the intensities of an object in image:

Then we can calculate the sum of the elements in the second object:

That is, however, not particularly efficient and may also be more complicated for other types of measurements. Therefore, a few measurements functions are defined that accept the array of object labels and the index of the object to be measured. For instance, calculating the sum of the intensities can be done by:

For large arrays and small objects, it is more efficient to call the measurement functions after slicing the array:

Alternatively, we can do the measurements for a number of labels with a single function call, returning a list of results. For instance, to measure the sum of the values of the background and the second object in our example, we give a list of labels:

The measurement functions described below all support the index parameter to indicate which object(s) should be measured. The default value of index is None. This indicates that all elements where the label is larger than zero should be treated as a single object and measured. Thus, in this case the labels array is treated as a mask defined by the elements that are larger than zero. If index is a number or a sequence of numbers it gives the labels of the objects that are measured. If index is a sequence, a list of the results is returned. Functions that return more than one result return their result as a tuple if index is a single number, or as a tuple of lists if index is a sequence.

The sum function calculates the sum of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The mean function calculates the mean of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The variance function calculates the variance of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The standard_deviation function calculates the standard deviation of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The minimum function calculates the minimum of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The maximum function calculates the maximum of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The minimum_position function calculates the position of the minimum of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The maximum_position function calculates the position of the maximum of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The extrema function calculates the minimum, the maximum, and their positions, of the elements of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation. The result is a tuple giving the minimum, the maximum, the position of the minimum, and the position of the maximum. The result is the same as a tuple formed by the results of the functions minimum, maximum, minimum_position, and maximum_position that are described above.

The center_of_mass function calculates the center of mass of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation.

The histogram function calculates a histogram of the object with label(s) given by index, using the labels array for the object labels. If index is None, all elements with a non-zero label value are treated as a single object. If label is None, all elements of input are used in the calculation. Histograms are defined by their minimum (min), maximum (max), and the number of bins (bins). They are returned as 1-D arrays of type numpy.int32.

A few functions in scipy.ndimage take a callback argument. This can be either a python function or a scipy.LowLevelCallable containing a pointer to a C function. Using a C function will generally be more efficient, since it avoids the overhead of calling a python function on many elements of an array. To use a C function, you must write a C extension that contains the callback function and a Python function that returns a scipy.LowLevelCallable containing a pointer to the callback.

An example of a function that supports callbacks is geometric_transform, which accepts a callback function that defines a mapping from all output coordinates to corresponding coordinates in the input array. Consider the following python example, which uses geometric_transform to implement a shift function.

We can also implement the callback function with the following C code:

More information on writing Python extension modules can be found here. If the C code is in the file example.c, then it can be compiled after adding it to meson.build (see examples inside meson.build files) and follow what’s there. After that is done, running the script:

produces the same result as the original python script.

In the C version, _transform is the callback function and the parameters output_coordinates and input_coordinates play the same role as they do in the python version, while output_rank and input_rank provide the equivalents of len(output_coordinates) and len(input_coordinates). The variable shift is passed through user_data instead of extra_arguments. Finally, the C callback function returns an integer status, which is one upon success and zero otherwise.

The function py_transform wraps the callback function in a PyCapsule. The main steps are:

Initialize a PyCapsule. The first argument is a pointer to the callback function.

The second argument is the function signature, which must match exactly the one expected by ndimage.

Above, we used scipy.LowLevelCallable to specify user_data that we generated with ctypes.

A different approach would be to supply the data in the capsule context, that can be set by PyCapsule_SetContext and omit specifying user_data in scipy.LowLevelCallable. However, in this approach we would need to deal with allocation/freeing of the data — freeing the data after the capsule has been destroyed can be done by specifying a non-NULL callback function in the third argument of PyCapsule_New.

C callback functions for ndimage all follow this scheme. The next section lists the ndimage functions that accept a C callback function and gives the prototype of the function.

The functions that support low-level callback arguments are:

generic_filter, generic_filter1d, geometric_transform

Below, we show alternative ways to write the code, using Numba, Cython, ctypes, or cffi instead of writing wrapper code in C.

Numba provides a way to write low-level functions easily in Python. We can write the above using Numba as:

Functionally the same code as above can be written in Cython with somewhat less boilerplate as follows:

With cffi, you can interface with a C function residing in a shared library (DLL). First, we need to write the shared library, which we do in C — this example is for Linux/OSX:

The Python code calling the library is:

You can find more information in the cffi documentation.

With ctypes, the C code and the compilation of the so/DLL is as for cffi above. The Python code is different:

You can find more information in the ctypes documentation.

M. Unser, “Splines: A Perfect Fit for Signal and Image Processing,” IEEE Signal Processing Magazine, vol. 16, no. 6, pp. 22-38, November 1999.

G. Borgefors, “Distance transformations in arbitrary dimensions.”, Computer Vision, Graphics, and Image Processing, 27:321-345, 1984.

C. R. Maurer, Jr., R. Qi, and V. Raghavan, “A linear time algorithm for computing exact euclidean distance transforms of binary images in arbitrary dimensions.” IEEE Trans. PAMI 25, 265-270, 2003.

A. X. Falcão, J. Stolfi, and R. A. Lotufo. “The image foresting transform: Theory, algorithms, and applications.” IEEE Trans. PAMI 26, 19-29. 2004.

T. Briand and P. Monasse, “Theory and Practice of Image B-Spline Interpolation”, Image Processing On Line, 8, pp. 99–141, 2018. https://doi.org/10.5201/ipol.2018.221

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.ndimage import correlate
>>> import numpy as np
>>> correlate(np.arange(10), [1, 2.5])
array([ 0,  2,  6,  9, 13, 16, 20, 23, 27, 30])
>>> correlate(np.arange(10), [1, 2.5], output=np.float64)
array([  0. ,   2.5,   6. ,   9.5,  13. ,  16.5,  20. ,  23.5,  27. ,  30.5])
```

Example 2 (json):
```json
>>> footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
>>> footprint
array([[0, 1, 0],
       [1, 1, 1],
       [0, 1, 0]])
```

Example 3 (sql):
```sql
>>> from scipy.ndimage import correlate1d
>>> a = [0, 0, 0, 1, 0, 0, 0]
>>> correlate1d(a, [1, 1, 1])
array([0, 0, 1, 1, 1, 0, 0])
```

Example 4 (unknown):
```unknown
>>> a = [0, 0, 0, 1, 0, 0, 0]
>>> correlate1d(a, [1, 1, 1], origin = -1)
array([0, 1, 1, 1, 0, 0, 0])
```

---

## helmert#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.helmert.html

**Contents:**
- helmert#

Create an Helmert matrix of order n.

This has applications in statistics, compositional or simplicial analysis, and in Aitchison geometry.

The size of the array to create.

If True the (n, n) ndarray will be returned. Otherwise the submatrix that does not include the first row will be returned. Default: False.

The Helmert matrix. The shape is (n, n) or (n-1, n) depending on the full argument.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import helmert
>>> helmert(5, full=True)
array([[ 0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ],
       [ 0.70710678, -0.70710678,  0.        ,  0.        ,  0.        ],
       [ 0.40824829,  0.40824829, -0.81649658,  0.        ,  0.        ],
       [ 0.28867513,  0.28867513,  0.28867513, -0.8660254 ,  0.        ],
       [ 0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 , -0.89442719]])
```

---

## tplquad#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.tplquad.html

**Contents:**
- tplquad#

Compute a triple (definite) integral.

Return the triple integral of func(z, y, x) from x = a..b, y = gfun(x)..hfun(x), and z = qfun(x,y)..rfun(x,y).

A Python function or method of at least three variables in the order (z, y, x).

The limits of integration in x: a < b

The lower boundary curve in y which is a function taking a single floating point argument (x) and returning a floating point result or a float indicating a constant boundary curve.

The upper boundary curve in y (same requirements as gfun).

The lower boundary surface in z. It must be a function that takes two floats in the order (x, y) and returns a float or a float indicating a constant boundary surface.

The upper boundary surface in z. (Same requirements as qfun.)

Extra arguments to pass to func.

Absolute tolerance passed directly to the innermost 1-D quadrature integration. Default is 1.49e-8.

Relative tolerance of the innermost 1-D integrals. Default is 1.49e-8.

The resultant integral.

An estimate of the error.

Adaptive quadrature using QUADPACK

Fixed-order Gaussian quadrature

N-dimensional integrals

Integrators for sampled data

Integrators for sampled data

For coefficients and roots of orthogonal polynomials

For valid results, the integral must converge; behavior for divergent integrals is not guaranteed.

Details of QUADPACK level routines

quad calls routines from the FORTRAN library QUADPACK. This section provides details on the conditions for each routine to be called and a short description of each routine. For each level of integration, qagse is used for finite limits or qagie is used, if either limit (or both!) are infinite. The following provides a short description from [1] for each routine.

is an integrator based on globally adaptive interval subdivision in connection with extrapolation, which will eliminate the effects of integrand singularities of several types. The integration is is performed using a 21-point Gauss-Kronrod quadrature within each subinterval.

handles integration over infinite intervals. The infinite range is mapped onto a finite interval and subsequently the same strategy as in QAGS is applied.

Piessens, Robert; de Doncker-Kapenga, Elise; Überhuber, Christoph W.; Kahaner, David (1983). QUADPACK: A subroutine package for automatic integration. Springer-Verlag. ISBN 978-3-540-12553-2.

Compute the triple integral of x * y * z, over x ranging from 1 to 2, y ranging from 2 to 3, z ranging from 0 to 1. That is, \(\int^{x=2}_{x=1} \int^{y=3}_{y=2} \int^{z=1}_{z=0} x y z \,dz \,dy \,dx\).

Calculate \(\int^{x=1}_{x=0} \int^{y=1-2x}_{y=0} \int^{z=1-x-2y}_{z=0} x y z \,dz \,dy \,dx\). Note: qfun/rfun takes arguments in the order (x, y), even though f takes arguments in the order (z, y, x).

Calculate \(\int^{x=1}_{x=0} \int^{y=1}_{y=0} \int^{z=1}_{z=0} a x y z \,dz \,dy \,dx\) for \(a=1, 3\).

Compute the three-dimensional Gaussian Integral, which is the integral of the Gaussian function \(f(x,y,z) = e^{-(x^{2} + y^{2} + z^{2})}\), over \((-\infty,+\infty)\). That is, compute the integral \(\iiint^{+\infty}_{-\infty} e^{-(x^{2} + y^{2} + z^{2})} \,dz \,dy\,dx\).

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import integrate
>>> f = lambda z, y, x: x*y*z
>>> integrate.tplquad(f, 1, 2, 2, 3, 0, 1)
(1.8749999999999998, 3.3246447942574074e-14)
```

Example 2 (json):
```json
>>> f = lambda z, y, x: x*y*z
>>> integrate.tplquad(f, 0, 1, 0, lambda x: 1-2*x, 0, lambda x, y: 1-x-2*y)
(0.05416666666666668, 2.1774196738157757e-14)
```

Example 3 (unknown):
```unknown
>>> f = lambda z, y, x, a: a*x*y*z
>>> integrate.tplquad(f, 0, 1, 0, 1, 0, 1, args=(1,))
    (0.125, 5.527033708952211e-15)
>>> integrate.tplquad(f, 0, 1, 0, 1, 0, 1, args=(3,))
    (0.375, 1.6581101126856635e-14)
```

Example 4 (unknown):
```unknown
>>> f = lambda x, y, z: np.exp(-(x ** 2 + y ** 2 + z ** 2))
>>> integrate.tplquad(f, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
    (5.568327996830833, 4.4619078828029765e-08)
```

---

## rfftn#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftn.html

**Contents:**
- rfftn#

Compute the N-D discrete Fourier Transform for real input.

This function computes the N-D discrete Fourier Transform over any number of axes in an M-D real array by means of the Fast Fourier Transform (FFT). By default, all axes are transformed, with the real transform performed over the last axis, while the remaining transforms are complex.

Input array, taken to be real.

Shape (length along each transformed axis) to use from the input. (s[0] refers to axis 0, s[1] to axis 1, etc.). The final element of s corresponds to n for rfft(x, n), while for the remaining axes, it corresponds to n for fft(x, n). Along any axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros. if s is not given, the shape of the input along the axes specified by axes is used.

Axes over which to compute the FFT. If not given, the last len(s) axes are used, or all axes if s is also not specified.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axes indicated by axes, or by a combination of s and x, as explained in the parameters section above. The length of the last axis transformed will be s[-1]//2+1, while the remaining transformed axes will have lengths according to s, or unchanged from the input.

If s and axes have different length.

If an element of axes is larger than the number of axes of x.

The inverse of rfftn, i.e., the inverse of the N-D FFT of real input.

The 1-D FFT, with definitions and conventions used.

The 1-D FFT of real input.

The 2-D FFT of real input.

The transform for real input is performed over the last transformation axis, as by rfft, then the transform over the remaining axes is performed as by fftn. The order of the output is as for rfft for the final transformation axis, and as for fftn for the remaining transformation axes.

See fft for details, definitions and conventions used.

**Examples:**

Example 1 (json):
```json
>>> import scipy.fft
>>> import numpy as np
>>> x = np.ones((2, 2, 2))
>>> scipy.fft.rfftn(x)
array([[[8.+0.j,  0.+0.j], # may vary
        [0.+0.j,  0.+0.j]],
       [[0.+0.j,  0.+0.j],
        [0.+0.j,  0.+0.j]]])
```

Example 2 (json):
```json
>>> scipy.fft.rfftn(x, axes=(2, 0))
array([[[4.+0.j,  0.+0.j], # may vary
        [4.+0.j,  0.+0.j]],
       [[0.+0.j,  0.+0.j],
        [0.+0.j,  0.+0.j]]])
```

---

## irfft#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfft.html

**Contents:**
- irfft#

Computes the inverse of rfft.

This function computes the inverse of the 1-D n-point discrete Fourier Transform of real input computed by rfft. In other words, irfft(rfft(x), len(x)) == x to within numerical accuracy. (See Notes below for why len(a) is necessary here.)

The input is expected to be in the form returned by rfft, i.e., the real zero-frequency term followed by the complex positive frequency terms in order of increasing frequency. Since the discrete Fourier Transform of real input is Hermitian-symmetric, the negative frequency terms are taken to be the complex conjugates of the corresponding positive frequency terms.

Length of the transformed axis of the output. For n output points, n//2+1 input points are necessary. If the input is longer than this, it is cropped. If it is shorter than this, it is padded with zeros. If n is not given, it is taken to be 2*(m-1), where m is the length of the input along the axis specified by axis.

Axis over which to compute the inverse FFT. If not given, the last axis is used.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified. The length of the transformed axis is n, or, if n is not given, 2*(m-1) where m is the length of the transformed axis of the input. To get an odd number of output points, n must be specified.

If axis is larger than the last axis of x.

The 1-D FFT of real input, of which irfft is inverse.

The inverse of the 2-D FFT of real input.

The inverse of the N-D FFT of real input.

Returns the real valued n-point inverse discrete Fourier transform of x, where x contains the non-negative frequency terms of a Hermitian-symmetric sequence. n is the length of the result, not the input.

If you specify an n such that a must be zero-padded or truncated, the extra/removed values will be added/removed at high frequencies. One can thus resample a series to m points via Fourier interpolation by: a_resamp = irfft(rfft(a), m).

The default value of n assumes an even output length. By the Hermitian symmetry, the last imaginary component must be 0 and so is ignored. To avoid losing information, the correct length of the real input must be given.

Notice how the last term in the input to the ordinary ifft is the complex conjugate of the second term, and the output has zero imaginary part everywhere. When calling irfft, the negative frequencies are not specified, and the output array is purely real.

**Examples:**

Example 1 (python):
```python
>>> import scipy.fft
>>> scipy.fft.ifft([1, -1j, -1, 1j])
array([0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]) # may vary
>>> scipy.fft.irfft([1, -1j, -1])
array([0.,  1.,  0.,  0.])
```

---

## romb#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html

**Contents:**
- romb#

Romberg integration using samples of a function.

A vector of 2**k + 1 equally-spaced samples of a function.

The sample spacing. Default is 1.

The axis along which to integrate. Default is -1 (last axis).

When y is a single 1-D array, then if this argument is True print the table showing Richardson extrapolation from the samples. Default is False.

The integrated result for axis.

adaptive quadrature using QUADPACK

fixed-order Gaussian quadrature

integrators for sampled data

cumulative integration for sampled data

**Examples:**

Example 1 (python):
```python
>>> from scipy import integrate
>>> import numpy as np
>>> x = np.arange(10, 14.25, 0.25)
>>> y = np.arange(3, 12)
```

Example 2 (unknown):
```unknown
>>> integrate.romb(y)
56.0
```

Example 3 (unknown):
```unknown
>>> y = np.sin(np.power(x, 2.5))
>>> integrate.romb(y)
-0.742561336672229
```

Example 4 (bash):
```bash
>>> integrate.romb(y, show=True)
Richardson Extrapolation Table for Romberg Integration
======================================================
-0.81576
 4.63862  6.45674
-1.10581 -3.02062 -3.65245
-2.57379 -3.06311 -3.06595 -3.05664
-1.34093 -0.92997 -0.78776 -0.75160 -0.74256
======================================================
-0.742561336672229  # may vary
```

---

## dft#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.dft.html

**Contents:**
- dft#

Discrete Fourier transform matrix.

Create the matrix that computes the discrete Fourier transform of a sequence [1]. The nth primitive root of unity used to generate the matrix is exp(-2*pi*i/n), where i = sqrt(-1).

Size the matrix to create.

Must be None, ‘sqrtn’, or ‘n’. If scale is ‘sqrtn’, the matrix is divided by sqrt(n). If scale is ‘n’, the matrix is divided by n. If scale is None (the default), the matrix is not normalized, and the return value is simply the Vandermonde matrix of the roots of unity.

When scale is None, multiplying a vector by the matrix returned by dft is mathematically equivalent to (but much less efficient than) the calculation performed by scipy.fft.fft.

Added in version 0.14.0.

“DFT matrix”, https://en.wikipedia.org/wiki/DFT_matrix

Verify that m @ x is the same as fft(x).

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import dft
>>> np.set_printoptions(precision=2, suppress=True)  # for compact output
>>> m = dft(5)
>>> m
array([[ 1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ],
       [ 1.  +0.j  ,  0.31-0.95j, -0.81-0.59j, -0.81+0.59j,  0.31+0.95j],
       [ 1.  +0.j  , -0.81-0.59j,  0.31+0.95j,  0.31-0.95j, -0.81+0.59j],
       [ 1.  +0.j  , -0.81+0.59j,  0.31-0.95j,  0.31+0.95j, -0.81-0.59j],
       [ 1.  +0.j  ,  0.31+0.95j, -0.81+0.59j, -0.81-0.59j,  0.31-0.95j]])
>>> x = np.array([1, 2, 3, 0, 3])
>>> m @ x  # Compute the DFT of x
array([ 9.  +0.j  ,  0.12-0.81j, -2.12+3.44j, -2.12-3.44j,  0.12+0.81j])
```

Example 2 (sql):
```sql
>>> from scipy.fft import fft
>>> fft(x)     # Same result as m @ x
array([ 9.  +0.j  ,  0.12-0.81j, -2.12+3.44j, -2.12-3.44j,  0.12+0.81j])
```

---

## Optimization and root finding (scipy.optimize)#

**URL:** https://docs.scipy.org/doc/scipy/reference/optimize.html

**Contents:**
- Optimization and root finding (scipy.optimize)#
- Optimization#
  - Scalar functions optimization#
  - Local (multivariate) optimization#
  - Global optimization#
- Least-squares and curve fitting#
  - Nonlinear least-squares#
  - Linear least-squares#
  - Curve fitting#
- Root finding#

SciPy optimize provides functions for minimizing (or maximizing) objective functions, possibly subject to constraints. It includes solvers for nonlinear problems (with support for both local and global optimization algorithms), linear programming, constrained and nonlinear least-squares, root finding, and curve fitting.

Common functions and objects, shared across different solvers, are:

show_options([solver, method, disp])

Show documentation for additional options of optimization solvers.

Represents the optimization result.

General warning for scipy.optimize.

minimize_scalar(fun[, bracket, bounds, ...])

Local minimization of scalar function of one variable.

The minimize_scalar function supports the following methods:

minimize(fun, x0[, args, method, jac, hess, ...])

Minimization of scalar function of one or more variables.

The minimize function supports the following methods:

Constraints are passed to minimize function as a single object or as a list of objects from the following classes:

NonlinearConstraint(fun, lb, ub[, jac, ...])

Nonlinear constraint on the variables.

LinearConstraint(A[, lb, ub, keep_feasible])

Linear constraint on the variables.

Simple bound constraints are handled separately and there is a special class for them:

Bounds([lb, ub, keep_feasible])

Bounds constraint on the variables.

Quasi-Newton strategies implementing HessianUpdateStrategy interface can be used to approximate the Hessian in minimize function (available only for the ‘trust-constr’ method). Available quasi-Newton methods implementing this interface are:

BFGS([exception_strategy, min_curvature, ...])

Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.

SR1([min_denominator, init_scale])

Symmetric-rank-1 Hessian update strategy.

basinhopping(func, x0[, niter, T, stepsize, ...])

Find the global minimum of a function using the basin-hopping algorithm.

brute(func, ranges[, args, Ns, full_output, ...])

Minimize a function over a given range by brute force.

differential_evolution(func, bounds[, args, ...])

Finds the global minimum of a multivariate function.

shgo(func, bounds[, args, constraints, n, ...])

Finds the global minimum of a function using SHG optimization.

dual_annealing(func, bounds[, args, ...])

Find the global minimum of a function using Dual Annealing.

direct(func, bounds, *[, args, eps, maxfun, ...])

Finds the global minimum of a function using the DIRECT algorithm.

least_squares(fun, x0[, jac, bounds, ...])

Solve a nonlinear least-squares problem with bounds on the variables.

nnls(A, b, *[, maxiter, atol])

Solve argmin_x || Ax - b ||_2 for x>=0.

lsq_linear(A, b[, bounds, method, tol, ...])

Solve a linear least-squares problem with bounds on the variables.

isotonic_regression(y, *[, weights, increasing])

Nonparametric isotonic regression.

curve_fit(f, xdata, ydata[, p0, sigma, ...])

Use non-linear least squares to fit a function, f, to data.

root_scalar(f[, args, method, bracket, ...])

Find a root of a scalar function.

brentq(f, a, b[, args, xtol, rtol, maxiter, ...])

Find a root of a function in a bracketing interval using Brent's method.

brenth(f, a, b[, args, xtol, rtol, maxiter, ...])

Find a root of a function in a bracketing interval using Brent's method with hyperbolic extrapolation.

ridder(f, a, b[, args, xtol, rtol, maxiter, ...])

Find a root of a function in an interval using Ridder's method.

bisect(f, a, b[, args, xtol, rtol, maxiter, ...])

Find root of a function within an interval using bisection.

newton(func, x0[, fprime, args, tol, ...])

Find a root of a real or complex function using the Newton-Raphson (or secant or Halley's) method.

toms748(f, a, b[, args, k, xtol, rtol, ...])

Find a root using TOMS Algorithm 748 method.

RootResults(root, iterations, ...)

Represents the root finding result.

The root_scalar function supports the following methods:

The table below lists situations and appropriate methods, along with asymptotic convergence rates per iteration (and per function evaluation) for successful convergence to a simple root(*). Bisection is the slowest of them all, adding one bit of accuracy for each function evaluation, but is guaranteed to converge. The other bracketing methods all (eventually) increase the number of accurate bits by about 50% for every function evaluation. The derivative-based methods, all built on newton, can converge quite quickly if the initial value is close to the root. They can also be applied to functions defined on (a subset of) the complex plane.

scipy.optimize.cython_optimize – Typed Cython versions of root finding functions

fixed_point(func, x0[, args, xtol, maxiter, ...])

Find a fixed point of the function.

root(fun, x0[, args, method, jac, tol, ...])

Find a root of a vector function.

The root function supports the following methods:

milp(c, *[, integrality, bounds, ...])

Mixed-integer linear programming

linprog(c[, A_ub, b_ub, A_eq, b_eq, bounds, ...])

Linear programming: minimize a linear objective function subject to linear equality and inequality constraints.

The linprog function supports the following methods:

The simplex, interior-point, and revised simplex methods support callback functions, such as:

linprog_verbose_callback(res)

A sample callback function demonstrating the linprog callback interface.

linear_sum_assignment

Solve the linear sum assignment problem.

quadratic_assignment(A, B[, method, options])

Approximates solution to the quadratic assignment problem and the graph matching problem.

The quadratic_assignment function supports the following methods:

approx_fprime(xk, f[, epsilon])

Finite difference approximation of the derivatives of a scalar or vector-valued function.

check_grad(func, grad, x0, *args[, epsilon, ...])

Check the correctness of a gradient function by comparing it against a (forward) finite-difference approximation of the gradient.

bracket(func[, xa, xb, args, grow_limit, ...])

Bracket the minimum of a function.

line_search(f, myfprime, xk, pk[, gfk, ...])

Find alpha that satisfies strong Wolfe conditions.

LbfgsInvHessProduct(*args, **kwargs)

Linear operator for the L-BFGS approximate inverse Hessian.

HessianUpdateStrategy()

Interface for implementing Hessian update strategies.

The Rosenbrock function.

The derivative (i.e. gradient) of the Rosenbrock function.

The Hessian matrix of the Rosenbrock function.

rosen_hess_prod(x, p)

Product of the Hessian matrix of the Rosenbrock function with a vector.

The functions below are not recommended for use in new scripts; all of these methods are accessible via a newer, more consistent interfaces, provided by the interfaces above.

General-purpose multivariate methods:

fmin(func, x0[, args, xtol, ftol, maxiter, ...])

Minimize a function using the downhill simplex algorithm.

fmin_powell(func, x0[, args, xtol, ftol, ...])

Minimize a function using modified Powell's method.

fmin_cg(f, x0[, fprime, args, gtol, norm, ...])

Minimize a function using a nonlinear conjugate gradient algorithm.

fmin_bfgs(f, x0[, fprime, args, gtol, norm, ...])

Minimize a function using the BFGS algorithm.

fmin_ncg(f, x0, fprime[, fhess_p, fhess, ...])

Unconstrained minimization of a function using the Newton-CG method.

Constrained multivariate methods:

fmin_l_bfgs_b(func, x0[, fprime, args, ...])

Minimize a function func using the L-BFGS-B algorithm.

fmin_tnc(func, x0[, fprime, args, ...])

Minimize a function with variables subject to bounds, using gradient information in a truncated Newton algorithm.

fmin_cobyla(func, x0, cons[, args, ...])

Minimize a function using the Constrained Optimization By Linear Approximation (COBYLA) method.

fmin_slsqp(func, x0[, eqcons, f_eqcons, ...])

Minimize a function using Sequential Least Squares Programming

Univariate (scalar) minimization methods:

fminbound(func, x1, x2[, args, xtol, ...])

Bounded minimization for scalar functions.

brent(func[, args, brack, tol, full_output, ...])

Given a function of one variable and a possible bracket, return a local minimizer of the function isolated to a fractional precision of tol.

golden(func[, args, brack, tol, ...])

Return the minimizer of a function of one variable using the golden section method.

leastsq(func, x0[, args, Dfun, full_output, ...])

Minimize the sum of squares of a set of equations.

General nonlinear solvers:

fsolve(func, x0[, args, fprime, ...])

Find the roots of a function.

broyden1(F, xin[, iter, alpha, ...])

Find a root of a function, using Broyden's first Jacobian approximation.

broyden2(F, xin[, iter, alpha, ...])

Find a root of a function, using Broyden's second Jacobian approximation.

Exception raised when nonlinear solver fails to converge within the specified maxiter.

Large-scale nonlinear solvers:

newton_krylov(F, xin[, iter, rdiff, method, ...])

Find a root of a function, using Krylov approximation for inverse Jacobian.

anderson(F, xin[, iter, alpha, w0, M, ...])

Find a root of a function, using (extended) Anderson mixing.

BroydenFirst([alpha, reduction_method, max_rank])

Find a root of a function, using Broyden's first Jacobian approximation.

InverseJacobian(jacobian)

A simple wrapper that inverts the Jacobian using the solve method.

KrylovJacobian([rdiff, method, ...])

Find a root of a function, using Krylov approximation for inverse Jacobian.

Simple iteration solvers:

excitingmixing(F, xin[, iter, alpha, ...])

Find a root of a function, using a tuned diagonal Jacobian approximation.

linearmixing(F, xin[, iter, alpha, verbose, ...])

Find a root of a function, using a scalar Jacobian approximation.

diagbroyden(F, xin[, iter, alpha, verbose, ...])

Find a root of a function, using diagonal Broyden Jacobian approximation.

---

## Interpolation (scipy.interpolate)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/interpolate.html

**Contents:**
- Interpolation (scipy.interpolate)#
- Smoothing and approximation of data#

There are several general facilities available in SciPy for interpolation and smoothing for data in 1, 2, and higher dimensions. The choice of a specific interpolation routine depends on the data: whether it is one-dimensional, is given on a structured grid, or is unstructured. One other factor is the desired smoothness of the interpolator. In short, routines recommended for interpolation can be summarized as follows:

Alternatively, make_interp_spline(..., k=1)

monotone cubic spline

k=3 is equivalent to CubicSpline

kind=’nearest’, ‘previous’, ‘next’

nearest, linear, spline

N-D regular (rectilinear) grid

RegularGridInterpolator

method=’cubic’, ‘quintic’

NearestNDInterpolator

CloughTocher2DInterpolator

radial basis function

make_smoothing_spline

classic smoothing splines, GCV penalty

automated/semi-automated knot selection

unconstrained least squares spline fit

2D smoothing surfaces

Radial basis functions in N-D

Further details are given in the links below

---

## correlate#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.correlate.html

**Contents:**
- correlate#

Multidimensional correlation.

The array is correlated with the given kernel.

array of weights, same number of dimensions as input

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for mode or origin must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

The result of correlation of input with weights.

Convolve an image with a kernel.

Correlation is the process of moving a filter mask often referred to as kernel over the image and computing the sum of products at each location.

Define a kernel (weights) for correlation. In this example, it is for sum of center and up, down, left and right next elements.

We can calculate a correlation result: For example, element [2,2] is 7 + 11 + 12 + 13 + 17 = 60.

**Examples:**

Example 1 (python):
```python
>>> from scipy.ndimage import correlate
>>> import numpy as np
>>> input_img = np.arange(25).reshape(5,5)
>>> print(input_img)
[[ 0  1  2  3  4]
[ 5  6  7  8  9]
[10 11 12 13 14]
[15 16 17 18 19]
[20 21 22 23 24]]
```

Example 2 (unknown):
```unknown
>>> weights = [[0, 1, 0],
...            [1, 1, 1],
...            [0, 1, 0]]
```

Example 3 (json):
```json
>>> correlate(input_img, weights)
array([[  6,  10,  15,  20,  24],
    [ 26,  30,  35,  40,  44],
    [ 51,  55,  60,  65,  69],
    [ 76,  80,  85,  90,  94],
    [ 96, 100, 105, 110, 114]])
```

---

## inv#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html

**Contents:**
- inv#

Compute the inverse of a matrix.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Square matrix to be inverted.

Discard data in a (may improve performance). Default is False.

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Inverse of the matrix a.

If a is not square, or not 2D.

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> a = np.array([[1., 2.], [3., 4.]])
>>> linalg.inv(a)
array([[-2. ,  1. ],
       [ 1.5, -0.5]])
>>> np.dot(a, linalg.inv(a))
array([[ 1.,  0.],
       [ 0.,  1.]])
```

---

## fourier_uniform#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_uniform.html

**Contents:**
- fourier_uniform#

Multidimensional uniform fourier filter.

The array is multiplied with the Fourier transform of a box of given size.

The size of the box used for filtering. If a float, size is the same for all axes. If a sequence, size has to contain one value for each axis.

If n is negative (default), then the input is assumed to be the result of a complex fft. If n is larger than or equal to zero, the input is assumed to be the result of a real fft, and n gives the length of the array before transformation along the real transform direction.

The axis of the real transform.

If given, the result of filtering the input is placed in this array.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = datasets.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_uniform(input_, size=20)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
```

---

## Fourier Transforms (scipy.fft)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/fft.html

**Contents:**
- Fourier Transforms (scipy.fft)#
- Fast Fourier transforms#
  - 1-D discrete Fourier transforms#
  - 2- and N-D discrete Fourier transforms#
- Discrete Cosine Transforms#
  - Type I DCT#
  - Type II DCT#
  - Type III DCT#
  - Type IV DCT#
  - DCT and IDCT#

Fourier Transforms (scipy.fft)

Fast Fourier transforms

1-D discrete Fourier transforms

2- and N-D discrete Fourier transforms

Discrete Cosine Transforms

Discrete Sine Transforms

Fast Hankel Transform

Fourier analysis is a method for expressing a function as a sum of periodic components, and for recovering the signal from those components. When both the function and its Fourier transform are replaced with discretized counterparts, it is called the discrete Fourier transform (DFT). The DFT has become a mainstay of numerical computing in part because of a very fast algorithm for computing it, called the Fast Fourier Transform (FFT), which was known to Gauss (1805) and was brought to light in its current form by Cooley and Tukey [CT65]. Press et al. [NR07] provide an accessible introduction to Fourier analysis and its applications.

The FFT y[k] of length \(N\) of the length-\(N\) sequence x[n] is defined as

and the inverse transform is defined as follows

These transforms can be calculated by means of fft and ifft, respectively, as shown in the following example.

From the definition of the FFT it can be seen that

which corresponds to \(y[0]\). For N even, the elements \(y[1]...y[N/2-1]\) contain the positive-frequency terms, and the elements \(y[N/2]...y[N-1]\) contain the negative-frequency terms, in order of decreasingly negative frequency. For N odd, the elements \(y[1]...y[(N-1)/2]\) contain the positive-frequency terms, and the elements \(y[(N+1)/2]...y[N-1]\) contain the negative-frequency terms, in order of decreasingly negative frequency.

In case the sequence x is real-valued, the values of \(y[n]\) for positive frequencies is the conjugate of the values \(y[n]\) for negative frequencies (because the spectrum is symmetric). Typically, only the FFT corresponding to positive frequencies is plotted.

The example plots the FFT of the sum of two sines.

The FFT input signal is inherently truncated. This truncation can be modeled as multiplication of an infinite signal with a rectangular window function. In the spectral domain this multiplication becomes convolution of the signal spectrum with the window function spectrum, being of form \(\sin(x)/x\). This convolution is the cause of an effect called spectral leakage (see [WPW]). Windowing the signal with a dedicated window function helps mitigate spectral leakage. The example below uses a Blackman window from scipy.signal and shows the effect of windowing (the zero component of the FFT has been truncated for illustrative purposes).

In case the sequence x is complex-valued, the spectrum is no longer symmetric. To simplify working with the FFT functions, scipy provides the following two helper functions.

The function fftfreq returns the FFT sample frequency points.

In a similar spirit, the function fftshift allows swapping the lower and upper halves of a vector, so that it becomes suitable for display.

The example below plots the FFT of two complex exponentials; note the asymmetric spectrum.

The function rfft calculates the FFT of a real sequence and outputs the complex FFT coefficients \(y[n]\) for only half of the frequency range. The remaining negative frequency components are implied by the Hermitian symmetry of the FFT for a real input (y[n] = conj(y[-n])). In case of N being even: \([Re(y[0]) + 0j, y[1], ..., Re(y[N/2]) + 0j]\); in case of N being odd \([Re(y[0]) + 0j, y[1], ..., y[N/2]\). The terms shown explicitly as \(Re(y[k]) + 0j\) are restricted to be purely real since, by the hermitian property, they are their own complex conjugate.

The corresponding function irfft calculates the IFFT of the FFT coefficients with this special ordering.

Notice that the rfft of odd and even length signals are of the same shape. By default, irfft assumes the output signal should be of even length. And so, for odd signals, it will give the wrong result:

To recover the original odd-length signal, we must pass the output shape by the n parameter.

The functions fft2 and ifft2 provide 2-D FFT and IFFT, respectively. Similarly, fftn and ifftn provide N-D FFT, and IFFT, respectively.

For real-input signals, similarly to rfft, we have the functions rfft2 and irfft2 for 2-D real transforms; rfftn and irfftn for N-D real transforms.

The example below demonstrates a 2-D IFFT and plots the resulting (2-D) time-domain signals.

SciPy provides a DCT with the function dct and a corresponding IDCT with the function idct. There are 8 types of the DCT [WPC], [Mak]; however, only the first 4 types are implemented in scipy. “The” DCT generally refers to DCT type 2, and “the” Inverse DCT generally refers to DCT type 3. In addition, the DCT coefficients can be normalized differently (for most types, scipy provides None and ortho). Two parameters of the dct/idct function calls allow setting the DCT type and coefficient normalization.

For a single dimension array x, dct(x, norm=’ortho’) is equal to MATLAB dct(x).

SciPy uses the following definition of the unnormalized DCT-I (norm=None):

Note that the DCT-I is only supported for input size > 1.

SciPy uses the following definition of the unnormalized DCT-II (norm=None):

In case of the normalized DCT (norm='ortho'), the DCT coefficients \(y[k]\) are multiplied by a scaling factor f:

In this case, the DCT “base functions” \(\phi_k[n] = 2 f \cos \left({\pi(2n+1)k \over 2N} \right)\) become orthonormal:

SciPy uses the following definition of the unnormalized DCT-III (norm=None):

or, for norm='ortho':

SciPy uses the following definition of the unnormalized DCT-IV (norm=None):

or, for norm='ortho':

The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up to a factor of 2N. The orthonormalized DCT-III is exactly the inverse of the orthonormalized DCT- II. The function idct performs the mappings between the DCT and IDCT types, as well as the correct normalization.

The following example shows the relation between DCT and IDCT for different types and normalizations.

The DCT-II and DCT-III are each other’s inverses, so for an orthonormal transform we return back to the original signal.

Doing the same under default normalization, however, we pick up an extra scaling factor of \(2N=10\) since the forward transform is unnormalized.

For this reason, we should use the function idct using the same type for both, giving a correctly normalized result.

Analogous results can be seen for the DCT-I, which is its own inverse up to a factor of \(2(N-1)\).

And for the DCT-IV, which is also its own inverse up to a factor of \(2N\).

The DCT exhibits the “energy compaction property”, meaning that for many signals only the first few DCT coefficients have significant magnitude. Zeroing out the other coefficients leads to a small reconstruction error, a fact which is exploited in lossy signal compression (e.g. JPEG compression).

The example below shows a signal x and two reconstructions (\(x_{20}\) and \(x_{15}\)) from the signal’s DCT coefficients. The signal \(x_{20}\) is reconstructed from the first 20 DCT coefficients, \(x_{15}\) is reconstructed from the first 15 DCT coefficients. It can be seen that the relative error of using 20 coefficients is still very small (~0.1%), but provides a five-fold compression rate.

SciPy provides a DST [Mak] with the function dst and a corresponding IDST with the function idst.

There are, theoretically, 8 types of the DST for different combinations of even/odd boundary conditions and boundary offsets [WPS], only the first 4 types are implemented in scipy.

DST-I assumes the input is odd around n=-1 and n=N. SciPy uses the following definition of the unnormalized DST-I (norm=None):

Note also that the DST-I is only supported for input size > 1. The (unnormalized) DST-I is its own inverse, up to a factor of 2(N+1).

DST-II assumes the input is odd around n=-1/2 and even around n=N. SciPy uses the following definition of the unnormalized DST-II (norm=None):

DST-III assumes the input is odd around n=-1 and even around n=N-1. SciPy uses the following definition of the unnormalized DST-III (norm=None):

SciPy uses the following definition of the unnormalized DST-IV (norm=None):

or, for norm='ortho':

The following example shows the relation between DST and IDST for different types and normalizations.

The DST-II and DST-III are each other’s inverses, so for an orthonormal transform we return back to the original signal.

Doing the same under default normalization, however, we pick up an extra scaling factor of \(2N=10\) since the forward transform is unnormalized.

For this reason, we should use the function idst using the same type for both, giving a correctly normalized result.

Analogous results can be seen for the DST-I, which is its own inverse up to a factor of \(2(N-1)\).

And for the DST-IV, which is also its own inverse up to a factor of \(2N\).

SciPy provides the functions fht and ifht to perform the Fast Hankel Transform (FHT) and its inverse (IFHT) on logarithmically-spaced input arrays.

The FHT is the discretised version of the continuous Hankel transform defined by [Ham00]

with \(J_{\mu}\) the Bessel function of order \(\mu\). Under a change of variables \(r \to \log r\), \(k \to \log k\), this becomes

which is a convolution in logarithmic space. The FHT algorithm uses the FFT to perform this convolution on discrete input data.

Care must be taken to minimise numerical ringing due to the circular nature of FFT convolution. To ensure that the low-ringing condition [Ham00] holds, the output array can be slightly shifted by an offset computed using the fhtoffset function.

Cooley, James W., and John W. Tukey, 1965, “An algorithm for the machine calculation of complex Fourier series,” Math. Comput. 19: 297-301.

Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P., 2007, Numerical Recipes: The Art of Scientific Computing, ch. 12-13. Cambridge Univ. Press, Cambridge, UK.

J. Makhoul, 1980, ‘A Fast Cosine Transform in One and Two Dimensions’, IEEE Transactions on acoustics, speech and signal processing vol. 28(1), pp. 27-34, DOI:10.1109/TASSP.1980.1163351

A. J. S. Hamilton, 2000, “Uncorrelated modes of the non-linear power spectrum”, MNRAS, 312, 257. DOI:10.1046/j.1365-8711.2000.03071.x

https://en.wikipedia.org/wiki/Window_function

https://en.wikipedia.org/wiki/Discrete_cosine_transform

https://en.wikipedia.org/wiki/Discrete_sine_transform

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.fft import fft, ifft
>>> import numpy as np
>>> x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
>>> y = fft(x)
>>> y
array([ 4.5       +0.j        ,  2.08155948-1.65109876j,
       -1.83155948+1.60822041j, -1.83155948-1.60822041j,
        2.08155948+1.65109876j])
>>> yinv = ifft(y)
>>> yinv
array([ 1.0+0.j,  2.0+0.j,  1.0+0.j, -1.0+0.j,  1.5+0.j])
```

Example 2 (unknown):
```unknown
>>> np.sum(x)
4.5
```

Example 3 (sql):
```sql
>>> from scipy.fft import fft, fftfreq
>>> import numpy as np
>>> # Number of sample points
>>> N = 600
>>> # sample spacing
>>> T = 1.0 / 800.0
>>> x = np.linspace(0.0, N*T, N, endpoint=False)
>>> y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
>>> yf = fft(y)
>>> xf = fftfreq(N, T)[:N//2]
>>> import matplotlib.pyplot as plt
>>> plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
>>> plt.grid()
>>> plt.show()
```

Example 4 (sql):
```sql
>>> from scipy.fft import fft, fftfreq
>>> import numpy as np
>>> # Number of sample points
>>> N = 600
>>> # sample spacing
>>> T = 1.0 / 800.0
>>> x = np.linspace(0.0, N*T, N, endpoint=False)
>>> y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
>>> yf = fft(y)
>>> from scipy.signal.windows import blackman
>>> w = blackman(N)
>>> ywf = fft(y*w)
>>> xf = fftfreq(N, T)[:N//2]
>>> import matplotlib.pyplot as plt
>>> plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
>>> plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
>>> plt.legend(['FFT', 'FFT w. window'])
>>> plt.grid()
>>> plt.show()
```

---

## mmread#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html

**Contents:**
- mmread#

Reads the contents of a Matrix Market file-like ‘source’ into a matrix.

Matrix Market filename (extensions .mtx, .mtz.gz) or open file-like object.

If True, return sparse coo_matrix. Otherwise return coo_array.

Dense or sparse array depending on the matrix format in the Matrix Market file.

Changed in version 1.12.0: C++ implementation.

mmread(source) returns the data as sparse array in COO format.

This method is threaded. The default number of threads is equal to the number of CPUs in the system. Use threadpoolctl to override:

**Examples:**

Example 1 (python):
```python
>>> from io import StringIO
>>> from scipy.io import mmread
```

Example 2 (unknown):
```unknown
>>> text = '''%%MatrixMarket matrix coordinate real general
...  5 5 7
...  2 3 1.0
...  3 4 2.0
...  3 5 3.0
...  4 1 4.0
...  4 2 5.0
...  4 3 6.0
...  4 4 7.0
... '''
```

Example 3 (jsx):
```jsx
>>> m = mmread(StringIO(text), spmatrix=False)
>>> m
<COOrdinate sparse array of dtype 'float64'
    with 7 stored elements and shape (5, 5)>
>>> m.toarray()
array([[0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 2., 3.],
       [4., 5., 6., 7., 0.],
       [0., 0., 0., 0., 0.]])
```

Example 4 (python):
```python
>>> import threadpoolctl
>>>
>>> with threadpoolctl.threadpool_limits(limits=2):
...     m = mmread(StringIO(text), spmatrix=False)
```

---

## Compressed Sparse Graph Routines (scipy.sparse.csgraph)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/csgraph.html

**Contents:**
- Compressed Sparse Graph Routines (scipy.sparse.csgraph)#
- Example: Word Ladders#

A Word Ladder is a word game invented by Lewis Carroll, in which players find paths between words by switching one letter at a time. For example, one can link “ape” and “man” in the following way:

Note that each step involves changing just one letter of the word. This is just one possible path from “ape” to “man”, but is it the shortest possible path? If we desire to find the shortest word-ladder path between two given words, the sparse graph submodule can help.

First, we need a list of valid words. Many operating systems have such a list built in. For example, on linux, a word list can often be found at one of the following locations:

Another easy source for words are the Scrabble word lists available at various sites around the internet (search with your favorite search engine). We’ll first create this list. The system word lists consist of a file with one word per line. The following should be modified to use the particular word list you have available:

We want to look at words of length 3, so let’s select just those words of the correct length. We’ll also eliminate words which start with upper-case (proper nouns) or contain non-alphanumeric characters, like apostrophes and hyphens. Finally, we’ll make sure everything is lower-case for comparison later:

Now we have a list of 586 valid three-letter words (the exact number may change depending on the particular list used). Each of these words will become a node in our graph, and we will create edges connecting the nodes associated with each pair of words which differs by only one letter.

There are efficient ways to do this, and inefficient ways to do this. To do this as efficiently as possible, we’re going to use some sophisticated numpy array manipulation:

We have an array where each entry is three unicode characters long. We’d like to find all pairs where exactly one character is different. We’ll start by converting each word to a 3-D vector:

Now, we’ll use the Hamming distance between each point to determine which pairs of words are connected. The Hamming distance measures the fraction of entries between two vectors which differ: any two words with a Hamming distance equal to \(1/N\), where \(N\) is the number of letters, are connected in the word ladder:

When comparing the distances, we don’t use an equality because this can be unstable for floating point values. The inequality produces the desired result, as long as no two entries of the word list are identical. Now, that our graph is set up, we’ll use a shortest path search to find the path between any two words in the graph:

We need to check that these match, because if the words are not in the list, that will not be the case. Now, all we need is to find the shortest path between these two indices in the graph. We’ll use Dijkstra’s algorithm, because it allows us to find the path for just one node:

So we see that the shortest path between “ape” and “man” contains only five steps. We can use the predecessors returned by the algorithm to reconstruct this path:

This is three fewer links than our initial example: the path from “ape” to “man” is only five steps.

Using other tools in the module, we can answer other questions. For example, are there three-letter words which are not linked in a word ladder? This is a question of connected components in the graph:

In this particular sample of three-letter words, there are 15 connected components: that is, 15 distinct sets of words with no paths between the sets. How many words are there in each of these sets? We can learn this from the list of components:

There is one large connected set and 14 smaller ones. Let’s look at the words in the smaller ones:

These are all the three-letter words which do not connect to others via a word ladder.

We might also be curious about which words are maximally separated. Which two words take the most links to connect? We can determine this by computing the matrix of all shortest paths. Note that, by convention, the distance between two non-connected points is reported to be infinity, so we’ll need to remove these before finding the maximum:

So, there is at least one pair of words which takes 13 steps to get from one to the other! Let’s determine which these are:

We see that there are two pairs of words which are maximally separated from each other: ‘imp’ and ‘ump’ on the one hand, and ‘ohm’ and ‘ohs’ on the other. We can find the connecting list in the same way as above:

This gives us the path we desired to see.

Word ladders are just one potential application of scipy’s fast graph algorithms for sparse matrices. Graph theory makes appearances in many areas of mathematics, data analysis, and machine learning. The sparse graph tools are flexible enough to handle many of these situations.

**Examples:**

Example 1 (unknown):
```unknown
/usr/share/dict
/var/lib/dict
```

Example 2 (typescript):
```typescript
>>> with open('/usr/share/dict/words') as f:
...    word_list = f.readlines()
>>> word_list = map(str.strip, word_list)
```

Example 3 (bash):
```bash
>>> word_list = [word for word in word_list if len(word) == 3]
>>> word_list = [word for word in word_list if word[0].islower()]
>>> word_list = [word for word in word_list if word.isalpha()]
>>> word_list = list(map(str.lower, word_list))
>>> len(word_list)
586    # may vary
```

Example 4 (typescript):
```typescript
>>> import numpy as np
>>> word_list = np.asarray(word_list)
>>> word_list.dtype   # these are unicode characters in Python 3
dtype('<U3')
>>> word_list.sort()  # sort for quick searching later
```

---

## toeplitz#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html

**Contents:**
- toeplitz#

Construct a Toeplitz matrix.

The Toeplitz matrix has constant diagonals, with c as its first column and r as its first row. If r is not given, r == conjugate(c) is assumed.

First column of the matrix.

First row of the matrix. If None, r = conjugate(c) is assumed; in this case, if c[0] is real, the result is a Hermitian matrix. r[0] is ignored; the first row of the returned matrix is [c[0], r[1:]].

Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not raveled. To preserve the existing behavior, ravel arguments before passing them to toeplitz.

The Toeplitz matrix. Dtype is the same as (c[0] + r[0]).dtype.

Solve a Toeplitz system.

The behavior when c or r is a scalar, or when c is complex and r is None, was changed in version 0.8.0. The behavior in previous versions was undocumented and is no longer supported.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import toeplitz
>>> toeplitz([1,2,3], [1,4,5,6])
array([[1, 4, 5, 6],
       [2, 1, 4, 5],
       [3, 2, 1, 4]])
>>> toeplitz([1.0, 2+3j, 4-1j])
array([[ 1.+0.j,  2.-3.j,  4.+1.j],
       [ 2.+3.j,  1.+0.j,  2.-3.j],
       [ 4.-1.j,  2.+3.j,  1.+0.j]])
```

---

## SciPy API#

**URL:** https://docs.scipy.org/doc/scipy/reference/index.html

**Contents:**
- SciPy API#
- Importing from SciPy#
- Guidelines for importing functions from SciPy#
- API definition#
- SciPy structure#

In Python, the distinction between what is the public API of a library and what are private implementation details is not always clear. Unlike in other languages like Java, it is possible in Python to access “private” functions or objects. Occasionally this may be convenient, but be aware that if you do so your code may break without warning in future releases. Some widely understood rules for what is and isn’t public in Python are:

Methods / functions / classes and module attributes whose names begin with a leading underscore are private.

If a class name begins with a leading underscore, none of its members are public, whether or not they begin with a leading underscore.

If a module name in a package begins with a leading underscore none of its members are public, whether or not they begin with a leading underscore.

If a module or package defines __all__, that authoritatively defines the public interface.

If a module or package doesn’t define __all__, then all names that don’t start with a leading underscore are public.

Reading the above guidelines one could draw the conclusion that every private module or object starts with an underscore. This is not the case; the presence of underscores do mark something as private, but the absence of underscores do not mark something as public.

In SciPy there are modules whose names don’t start with an underscore, but that should be considered private. To clarify which modules these are, we define below what the public API is for SciPy, and give some recommendations for how to import modules/functions/objects from SciPy.

Everything in the namespaces of SciPy submodules is public. In general in Python, it is recommended to make use of namespaces. For example, the function curve_fit (defined in scipy/optimize/_minpack_py.py) should be imported like this:

Or alternatively one could use the submodule as a namespace like so:

For scipy.io prefer the use of import scipy because io is also the name of a module in the Python stdlib.

In some cases, the public API is one level deeper. For example, the scipy.sparse.linalg module is public, and the functions it contains are not available in the scipy.sparse namespace. Sometimes it may result in more easily understandable code if functions are imported from one level deeper. For example, in the following it is immediately clear that lomax is a distribution if the second form is chosen:

In that case, the second form can be chosen if it is documented in the next section that the submodule in question is public. Of course you can still use:

SciPy is using a lazy loading mechanism which means that modules are only loaded in memory when you first try to access them.

Every submodule listed below is public. That means that these submodules are unlikely to be renamed or changed in an incompatible way, and if that is necessary, a deprecation warning will be raised for one SciPy release before the change is made.

scipy.cluster.hierarchy

scipy.linalg.cython_blas

scipy.linalg.cython_lapack

scipy.linalg.interpolative

scipy.optimize.cython_optimize

scipy.spatial.distance

scipy.spatial.transform

scipy.stats.contingency

scipy.stats.distributions

All SciPy modules should follow the following conventions. In the following, a SciPy module is defined as a Python package, say yyy, that is located in the scipy/ directory.

Ideally, each SciPy module should be as self-contained as possible. That is, it should have minimal dependencies on other packages or modules. Even dependencies on other SciPy modules should be kept to a minimum. A dependency on NumPy is of course assumed.

Directory yyy/ contains:

A file meson.build with build configuration for the submodule.

A directory tests/ that contains files test_<name>.py corresponding to modules yyy/<name>{.py,.so,/}.

Private modules should be prefixed with an underscore _, for instance yyy/_somemodule.py.

User-visible functions should have good documentation following the NumPy documentation style.

The __init__.py of the module should contain the main reference documentation in its docstring. This is connected to the Sphinx documentation under doc/ via Sphinx’s automodule directive.

The reference documentation should first give a categorized list of the contents of the module using autosummary:: directives, and after that explain points essential for understanding the use of the module.

Tutorial-style documentation with extensive examples should be separate and put under doc/source/tutorial/.

See the existing SciPy submodules for guidance.

**Examples:**

Example 1 (python):
```python
import scipy
result = scipy.optimize.curve_fit(...)
```

Example 2 (python):
```python
from scipy import optimize
result = optimize.curve_fit(...)
```

Example 3 (python):
```python
# first form
from scipy import stats
stats.lomax(...)

# second form
from scipy.stats import distributions
distributions.lomax(...)
```

Example 4 (markdown):
```markdown
import scipy
scipy.stats.lomax(...)
# or
scipy.stats.distributions.lomax(...)
```

---

## Special functions (scipy.special)#

**URL:** https://docs.scipy.org/doc/scipy/reference/special.html

**Contents:**
- Special functions (scipy.special)#
- Error handling#
- Available functions#
  - Airy functions#
  - Elliptic functions and integrals#
  - Bessel functions#
    - Zeros of Bessel functions#
    - Faster versions of common Bessel functions#
    - Integrals of Bessel functions#
    - Derivatives of Bessel functions#

Almost all of the functions below accept NumPy arrays as input arguments as well as single numbers. This means they follow broadcasting and automatic array-looping rules. Technically, they are NumPy universal functions. Functions which do not accept NumPy arrays are marked by a warning in the section description.

scipy.special.cython_special – Typed Cython versions of special functions

Errors are handled by returning NaNs or other appropriate values. Some of the special function routines can emit warnings or raise exceptions when an error occurs. By default this is disabled, except for memory allocation errors, which result in an exception being raised. To query and control the current error handling state the following functions are provided.

Get the current way of handling special-function errors.

Set how special-function errors are handled.

Context manager for special-function error handling.

SpecialFunctionWarning

Warning that can be emitted by special functions.

Exception that can be raised by special functions.

Airy functions and their derivatives.

Exponentially scaled Airy functions and their derivatives.

Compute nt zeros and values of the Airy function Ai and its derivative.

Compute nt zeros and values of the Airy function Bi and its derivative.

Integrals of Airy functions

Jacobian elliptic functions

Complete elliptic integral of the first kind.

Complete elliptic integral of the first kind around m = 1

ellipkinc(phi, m[, out])

Incomplete elliptic integral of the first kind

Complete elliptic integral of the second kind

ellipeinc(phi, m[, out])

Incomplete elliptic integral of the second kind

Degenerate symmetric elliptic integral.

elliprd(x, y, z[, out])

Symmetric elliptic integral of the second kind.

elliprf(x, y, z[, out])

Completely-symmetric elliptic integral of the first kind.

elliprg(x, y, z[, out])

Completely-symmetric elliptic integral of the second kind.

elliprj(x, y, z, p[, out])

Symmetric elliptic integral of the third kind.

Bessel function of the first kind of real order and complex argument.

Exponentially scaled Bessel function of the first kind of order v.

Bessel function of the second kind of integer order and real argument.

Bessel function of the second kind of real order and complex argument.

Exponentially scaled Bessel function of the second kind of real order.

Modified Bessel function of the second kind of integer order n

Modified Bessel function of the second kind of real order v

Exponentially scaled modified Bessel function of the second kind.

Modified Bessel function of the first kind of real order.

Exponentially scaled modified Bessel function of the first kind.

Hankel function of the first kind

hankel1e(v, z[, out])

Exponentially scaled Hankel function of the first kind

Hankel function of the second kind

hankel2e(v, z[, out])

Exponentially scaled Hankel function of the second kind

wright_bessel(a, b, x[, out])

Wright's generalized Bessel function.

log_wright_bessel(a, b, x[, out])

Natural logarithm of Wright's generalized Bessel function, see wright_bessel.

The following function does not accept NumPy arrays (it is not a universal function):

Jahnke-Emden Lambda function, Lambdav(x).

The following functions do not accept NumPy arrays (they are not universal functions):

Compute zeros of integer-order Bessel functions Jn and Jn'.

Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).

Compute zeros of integer-order Bessel functions Jn.

Compute zeros of integer-order Bessel function derivatives Jn'.

Compute zeros of integer-order Bessel function Yn(x).

Compute zeros of integer-order Bessel function derivatives Yn'(x).

y0_zeros(nt[, complex])

Compute nt zeros of Bessel function Y0(z), and derivative at each zero.

y1_zeros(nt[, complex])

Compute nt zeros of Bessel function Y1(z), and derivative at each zero.

y1p_zeros(nt[, complex])

Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.

Bessel function of the first kind of order 0.

Bessel function of the first kind of order 1.

Bessel function of the second kind of order 0.

Bessel function of the second kind of order 1.

Modified Bessel function of order 0.

Exponentially scaled modified Bessel function of order 0.

Modified Bessel function of order 1.

Exponentially scaled modified Bessel function of order 1.

Modified Bessel function of the second kind of order 0, \(K_0\).

Exponentially scaled modified Bessel function K of order 0

Modified Bessel function of the second kind of order 1, \(K_1(x)\).

Exponentially scaled modified Bessel function K of order 1

Integrals of Bessel functions of the first kind of order 0.

Integrals related to Bessel functions of the first kind of order 0.

Integrals of modified Bessel functions of order 0.

Integrals related to modified Bessel functions of order 0.

besselpoly(a, lmb, nu[, out])

Weighted integral of the Bessel function of the first kind.

Compute derivatives of Bessel functions of the first kind.

Compute derivatives of Bessel functions of the second kind.

Compute derivatives of real-order modified Bessel function Kv(z)

Compute derivatives of modified Bessel functions of the first kind.

Compute derivatives of Hankel function H1v(z) with respect to z.

Compute derivatives of Hankel function H2v(z) with respect to z.

spherical_jn(n, z[, derivative])

Spherical Bessel function of the first kind or its derivative.

spherical_yn(n, z[, derivative])

Spherical Bessel function of the second kind or its derivative.

spherical_in(n, z[, derivative])

Modified spherical Bessel function of the first kind or its derivative.

spherical_kn(n, z[, derivative])

Modified spherical Bessel function of the second kind or its derivative.

The following functions do not accept NumPy arrays (they are not universal functions):

Compute Ricatti-Bessel function of the first kind and its derivative.

Compute Ricatti-Bessel function of the second kind and its derivative.

modstruve(v, x[, out])

Modified Struve function.

Integral of the Struve function of order 0.

Integral related to the Struve function of order 0.

itmodstruve0(x[, out])

Integral of the modified Struve function of order 0.

scipy.stats: Friendly versions of these functions.

Binomial distribution cumulative distribution function.

bdtrc(k, n, p[, out])

Binomial distribution survival function.

bdtri(k, n, y[, out])

Inverse function to bdtr with respect to p.

bdtrik(y, n, p[, out])

Inverse function to bdtr with respect to k.

bdtrin(k, y, p[, out])

Inverse function to bdtr with respect to n.

btdtria(p, b, x[, out])

Inverse of betainc with respect to a.

btdtrib(a, p, x[, out])

Inverse of betainc with respect to b.

fdtr(dfn, dfd, x[, out])

F cumulative distribution function.

fdtrc(dfn, dfd, x[, out])

fdtri(dfn, dfd, p[, out])

The p-th quantile of the F-distribution.

fdtridfd(dfn, p, x[, out])

Inverse to fdtr vs dfd

Gamma distribution cumulative distribution function.

gdtrc(a, b, x[, out])

Gamma distribution survival function.

gdtria(p, b, x[, out])

Inverse of gdtr vs a.

gdtrib(a, p, x[, out])

Inverse of gdtr vs b.

gdtrix(a, b, p[, out])

Inverse of gdtr vs x.

nbdtr(k, n, p[, out])

Negative binomial cumulative distribution function.

nbdtrc(k, n, p[, out])

Negative binomial survival function.

nbdtri(k, n, y[, out])

Returns the inverse with respect to the parameter p of y = nbdtr(k, n, p), the negative binomial cumulative distribution function.

nbdtrik(y, n, p[, out])

Negative binomial percentile function.

nbdtrin(k, y, p[, out])

Inverse of nbdtr vs n.

ncfdtr(dfn, dfd, nc, f[, out])

Cumulative distribution function of the non-central F distribution.

ncfdtridfd(dfn, p, nc, f[, out])

Calculate degrees of freedom (denominator) for the noncentral F-distribution.

ncfdtridfn(p, dfd, nc, f[, out])

Calculate degrees of freedom (numerator) for the noncentral F-distribution.

ncfdtri(dfn, dfd, nc, p[, out])

Inverse with respect to f of the CDF of the non-central F distribution.

ncfdtrinc(dfn, dfd, p, f[, out])

Calculate non-centrality parameter for non-central F distribution.

nctdtr(df, nc, t[, out])

Cumulative distribution function of the non-central t distribution.

nctdtridf(p, nc, t[, out])

Calculate degrees of freedom for non-central t distribution.

nctdtrit(df, nc, p[, out])

Inverse cumulative distribution function of the non-central t distribution.

nctdtrinc(df, p, t[, out])

Calculate non-centrality parameter for non-central t distribution.

nrdtrimn(p, std, x[, out])

Calculate mean of normal distribution given other params.

nrdtrisd(mn, p, x[, out])

Calculate standard deviation of normal distribution given other params.

Cumulative distribution of the standard normal distribution.

Logarithm of Gaussian cumulative distribution function.

Inverse of log_ndtr vs x.

Poisson cumulative distribution function.

Poisson survival function

Inverse to pdtr vs k.

Student t distribution cumulative distribution function

stdtridf(p, t[, out])

Inverse of stdtr vs df

stdtrit(df, p[, out])

The p-th quantile of the student t distribution.

Chi square cumulative distribution function.

Chi square survival function.

Inverse to chdtrc with respect to x.

Inverse to chdtr with respect to v.

chndtr(x, df, nc[, out])

Non-central chi square cumulative distribution function

chndtridf(x, p, nc[, out])

Inverse to chndtr vs df

chndtrinc(x, df, p[, out])

Inverse to chndtr vs nc

chndtrix(p, df, nc[, out])

Inverse to chndtr vs x

Kolmogorov-Smirnov complementary cumulative distribution function

smirnovi(n, p[, out])

Complementary cumulative distribution (Survival Function) function of Kolmogorov distribution.

Inverse Survival Function of Kolmogorov distribution

boxcox(x, lmbda[, out])

Compute the Box-Cox transformation.

boxcox1p(x, lmbda[, out])

Compute the Box-Cox transformation of 1 + x.

inv_boxcox(y, lmbda[, out])

Compute the inverse of the Box-Cox transformation.

inv_boxcox1p(y, lmbda[, out])

Compute the inverse of the Box-Cox transformation.

Logit ufunc for ndarrays.

Logarithm of the logistic sigmoid function.

tklmbda(x, lmbda[, out])

Cumulative distribution function of the Tukey lambda distribution.

Elementwise function for computing entropy.

rel_entr(x, y[, out])

Elementwise function for computing relative entropy.

Elementwise function for computing Kullback-Leibler divergence.

huber(delta, r[, out])

pseudo_huber(delta, r[, out])

Pseudo-Huber loss function.

Logarithm of the absolute value of the gamma function.

Principal branch of the logarithm of the gamma function.

Sign of the gamma function.

gammainc(a, x[, out])

Regularized lower incomplete gamma function.

gammaincinv(a, y[, out])

Inverse to the regularized lower incomplete gamma function.

gammaincc(a, x[, out])

Regularized upper incomplete gamma function.

gammainccinv(a, y[, out])

Inverse of the regularized upper incomplete gamma function.

Natural logarithm of absolute value of beta function.

betainc(a, b, x[, out])

Regularized incomplete beta function.

betaincc(a, b, x[, out])

Complement of the regularized incomplete beta function.

betaincinv(a, b, y[, out])

Inverse of the regularized incomplete beta function.

betainccinv(a, b, y[, out])

Inverse of the complemented regularized incomplete beta function.

The digamma function.

Reciprocal of the gamma function.

Returns the log of multivariate gamma, also sometimes called the generalized gamma.

The digamma function.

Returns the error function of complex argument.

Complementary error function, 1 - erf(x).

Scaled complementary error function, exp(x**2) * erfc(x).

Imaginary error function, -i erf(i z).

Inverse of the error function.

Inverse of the complementary error function.

Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).

modfresnelp(x[, out])

Modified Fresnel positive integrals

modfresnelm(x[, out])

Modified Fresnel negative integrals

voigt_profile(x, sigma, gamma[, out])

The following functions do not accept NumPy arrays (they are not universal functions):

Compute the first nt zero in the first quadrant, ordered by absolute value.

Compute nt complex zeros of cosine Fresnel integral C(z).

Compute nt complex zeros of sine Fresnel integral S(z).

legendre_p(n, z, *[, diff_n])

Legendre polynomial of the first kind.

legendre_p_all(n, z, *[, diff_n])

All Legendre polynomials of the first kind up to the specified degree n.

assoc_legendre_p(n, m, z, *[, branch_cut, ...])

Associated Legendre polynomial of the first kind.

assoc_legendre_p_all(n, m, z, *[, ...])

All associated Legendre polynomials of the first kind up to the specified degree n and order m.

sph_legendre_p(n, m, theta, *[, diff_n])

Spherical Legendre polynomial of the first kind.

sph_legendre_p_all(n, m, theta, *[, diff_n])

All spherical Legendre polynomials of the first kind up to the specified degree n and order m.

sph_harm_y(n, m, theta, phi, *[, diff_n])

sph_harm_y_all(n, m, theta, phi, *[, diff_n])

All spherical harmonics up to the specified degree n and order m.

The following functions are in the process of being deprecated in favor of the above, which provide a more flexible and consistent interface.

Associated Legendre function of integer order and real degree.

sph_harm(m, n, theta, phi[, out])

Compute spherical harmonics.

clpmn(m, n, z[, type])

Associated Legendre function of the first kind for complex arguments.

Legendre function of the first kind.

Legendre function of the second kind.

Sequence of associated Legendre functions of the first kind.

Sequence of associated Legendre functions of the second kind.

ellip_harm(h2, k2, n, p, s[, signm, signn])

Ellipsoidal harmonic functions E^p_n(l)

ellip_harm_2(h2, k2, n, p, s)

Ellipsoidal harmonic functions F^p_n(l)

ellip_normal(h2, k2, n, p)

Ellipsoidal harmonic normalization constants gamma^p_n

The following functions evaluate values of orthogonal polynomials:

assoc_laguerre(x, n[, k])

Compute the generalized (associated) Laguerre polynomial of degree n and order k.

eval_legendre(n, x[, out])

Evaluate Legendre polynomial at a point.

eval_chebyt(n, x[, out])

Evaluate Chebyshev polynomial of the first kind at a point.

eval_chebyu(n, x[, out])

Evaluate Chebyshev polynomial of the second kind at a point.

eval_chebyc(n, x[, out])

Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a point.

eval_chebys(n, x[, out])

Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a point.

eval_jacobi(n, alpha, beta, x[, out])

Evaluate Jacobi polynomial at a point.

eval_laguerre(n, x[, out])

Evaluate Laguerre polynomial at a point.

eval_genlaguerre(n, alpha, x[, out])

Evaluate generalized Laguerre polynomial at a point.

eval_hermite(n, x[, out])

Evaluate physicist's Hermite polynomial at a point.

eval_hermitenorm(n, x[, out])

Evaluate probabilist's (normalized) Hermite polynomial at a point.

eval_gegenbauer(n, alpha, x[, out])

Evaluate Gegenbauer polynomial at a point.

eval_sh_legendre(n, x[, out])

Evaluate shifted Legendre polynomial at a point.

eval_sh_chebyt(n, x[, out])

Evaluate shifted Chebyshev polynomial of the first kind at a point.

eval_sh_chebyu(n, x[, out])

Evaluate shifted Chebyshev polynomial of the second kind at a point.

eval_sh_jacobi(n, p, q, x[, out])

Evaluate shifted Jacobi polynomial at a point.

The following functions compute roots and quadrature weights for orthogonal polynomials:

roots_legendre(n[, mu])

Gauss-Legendre quadrature.

roots_chebyt(n[, mu])

Gauss-Chebyshev (first kind) quadrature.

roots_chebyu(n[, mu])

Gauss-Chebyshev (second kind) quadrature.

roots_chebyc(n[, mu])

Gauss-Chebyshev (first kind) quadrature.

roots_chebys(n[, mu])

Gauss-Chebyshev (second kind) quadrature.

roots_jacobi(n, alpha, beta[, mu])

Gauss-Jacobi quadrature.

roots_laguerre(n[, mu])

Gauss-Laguerre quadrature.

roots_genlaguerre(n, alpha[, mu])

Gauss-generalized Laguerre quadrature.

roots_hermite(n[, mu])

Gauss-Hermite (physicist's) quadrature.

roots_hermitenorm(n[, mu])

Gauss-Hermite (statistician's) quadrature.

roots_gegenbauer(n, alpha[, mu])

Gauss-Gegenbauer quadrature.

roots_sh_legendre(n[, mu])

Gauss-Legendre (shifted) quadrature.

roots_sh_chebyt(n[, mu])

Gauss-Chebyshev (first kind, shifted) quadrature.

roots_sh_chebyu(n[, mu])

Gauss-Chebyshev (second kind, shifted) quadrature.

roots_sh_jacobi(n, p1, q1[, mu])

Gauss-Jacobi (shifted) quadrature.

The functions below, in turn, return the polynomial coefficients in orthopoly1d objects, which function similarly as numpy.poly1d. The orthopoly1d class also has an attribute weights, which returns the roots, weights, and total weights for the appropriate form of Gaussian quadrature. These are returned in an n x 3 array with roots in the first column, weights in the second column, and total weights in the final column. Note that orthopoly1d objects are converted to poly1d when doing arithmetic, and lose information of the original orthogonal polynomial.

Chebyshev polynomial of the first kind.

Chebyshev polynomial of the second kind.

Chebyshev polynomial of the first kind on \([-2, 2]\).

Chebyshev polynomial of the second kind on \([-2, 2]\).

jacobi(n, alpha, beta[, monic])

genlaguerre(n, alpha[, monic])

Generalized (associated) Laguerre polynomial.

Physicist's Hermite polynomial.

hermitenorm(n[, monic])

Normalized (probabilist's) Hermite polynomial.

gegenbauer(n, alpha[, monic])

Gegenbauer (ultraspherical) polynomial.

sh_legendre(n[, monic])

Shifted Legendre polynomial.

sh_chebyt(n[, monic])

Shifted Chebyshev polynomial of the first kind.

sh_chebyu(n[, monic])

Shifted Chebyshev polynomial of the second kind.

sh_jacobi(n, p, q[, monic])

Shifted Jacobi polynomial.

Computing values of high-order polynomials (around order > 20) using polynomial coefficients is numerically unstable. To evaluate polynomial values, the eval_* functions should be used instead.

hyp2f1(a, b, c, z[, out])

Gauss hypergeometric function 2F1(a, b; c; z)

hyp1f1(a, b, x[, out])

Confluent hypergeometric function 1F1.

hyperu(a, b, x[, out])

Confluent hypergeometric function U

Confluent hypergeometric limit function 0F1.

Parabolic cylinder function D

Parabolic cylinder function V

Parabolic cylinder function W.

The following functions do not accept NumPy arrays (they are not universal functions):

Parabolic cylinder functions Dv(x) and derivatives.

Parabolic cylinder functions Vv(x) and derivatives.

Parabolic cylinder functions Dn(z) and derivatives.

mathieu_a(m, q[, out])

Characteristic value of even Mathieu functions

mathieu_b(m, q[, out])

Characteristic value of odd Mathieu functions

The following functions do not accept NumPy arrays (they are not universal functions):

mathieu_even_coef(m, q)

Fourier coefficients for even Mathieu and modified Mathieu functions.

mathieu_odd_coef(m, q)

Fourier coefficients for even Mathieu and modified Mathieu functions.

The following return both function and first derivative:

mathieu_cem(m, q, x[, out])

Even Mathieu function and its derivative

mathieu_sem(m, q, x[, out])

Odd Mathieu function and its derivative

mathieu_modcem1(m, q, x[, out])

Even modified Mathieu function of the first kind and its derivative

mathieu_modcem2(m, q, x[, out])

Even modified Mathieu function of the second kind and its derivative

mathieu_modsem1(m, q, x[, out])

Odd modified Mathieu function of the first kind and its derivative

mathieu_modsem2(m, q, x[, out])

Odd modified Mathieu function of the second kind and its derivative

pro_ang1(m, n, c, x[, out])

Prolate spheroidal angular function of the first kind and its derivative

pro_rad1(m, n, c, x[, out])

Prolate spheroidal radial function of the first kind and its derivative

pro_rad2(m, n, c, x[, out])

Prolate spheroidal radial function of the second kind and its derivative

obl_ang1(m, n, c, x[, out])

Oblate spheroidal angular function of the first kind and its derivative

obl_rad1(m, n, c, x[, out])

Oblate spheroidal radial function of the first kind and its derivative

obl_rad2(m, n, c, x[, out])

Oblate spheroidal radial function of the second kind and its derivative.

pro_cv(m, n, c[, out])

Characteristic value of prolate spheroidal function

obl_cv(m, n, c[, out])

Characteristic value of oblate spheroidal function

Characteristic values for prolate spheroidal wave functions.

Characteristic values for oblate spheroidal wave functions.

The following functions require pre-computed characteristic value:

pro_ang1_cv(m, n, c, cv, x[, out])

Prolate spheroidal angular function pro_ang1 for precomputed characteristic value

pro_rad1_cv(m, n, c, cv, x[, out])

Prolate spheroidal radial function pro_rad1 for precomputed characteristic value

pro_rad2_cv(m, n, c, cv, x[, out])

Prolate spheroidal radial function pro_rad2 for precomputed characteristic value

obl_ang1_cv(m, n, c, cv, x[, out])

Oblate spheroidal angular function obl_ang1 for precomputed characteristic value

obl_rad1_cv(m, n, c, cv, x[, out])

Oblate spheroidal radial function obl_rad1 for precomputed characteristic value

obl_rad2_cv(m, n, c, cv, x[, out])

Oblate spheroidal radial function obl_rad2 for precomputed characteristic value

Kelvin functions as complex numbers

Compute nt zeros of all Kelvin functions.

Derivative of the Kelvin function ber.

Derivative of the Kelvin function bei.

Derivative of the Kelvin function ker.

Derivative of the Kelvin function kei.

The following functions do not accept NumPy arrays (they are not universal functions):

Compute nt zeros of the Kelvin function ber.

Compute nt zeros of the Kelvin function bei.

Compute nt zeros of the derivative of the Kelvin function ber.

Compute nt zeros of the derivative of the Kelvin function bei.

Compute nt zeros of the Kelvin function ker.

Compute nt zeros of the Kelvin function kei.

Compute nt zeros of the derivative of the Kelvin function ker.

Compute nt zeros of the derivative of the Kelvin function kei.

comb(N, k, *[, exact, repetition])

The number of combinations of N things taken k at a time.

Permutations of N things taken k at a time, i.e., k-permutations of N.

stirling2(N, K, *[, exact])

Generate Stirling number(s) of the second kind.

lambertw(z[, k, tol])

wrightomega(z[, out])

Wright Omega function.

Compute the arithmetic-geometric mean of a and b.

Bernoulli numbers B0..Bn (inclusive).

Binomial coefficient considered as a function of two real variables.

Periodic sinc function, also called the Dirichlet function.

Euler numbers E(0), E(1), ..., E(n).

Generalized exponential integral En.

Exponential integral E1.

Exponential integral Ei.

factorial(n[, exact, extend])

The factorial of a number or array of numbers.

factorial2(n[, exact, extend])

factorialk(n, k[, exact, extend])

Multifactorial of n of order k, n(!!...!).

Hyperbolic sine and cosine integrals.

Sine and cosine integrals.

Compute the softmax function.

log_softmax(x[, axis])

Compute the logarithm of the softmax function.

Spence's function, also known as the dilogarithm.

Riemann or Hurwitz zeta function.

Riemann zeta function minus 1.

softplus(x, **kwargs)

Compute the softplus function element-wise.

Element-wise cube root of x.

Compute 10**x element-wise.

Compute 2**x element-wise.

radian(d, m, s[, out])

Convert from degrees to radians.

Cosine of the angle x given in degrees.

Sine of the angle x given in degrees.

Tangent of angle x given in degrees.

Cotangent of the angle x given in degrees.

Calculates log(1 + x) for use when x is near zero.

cos(x) - 1 for use when x is near zero.

Round to the nearest integer.

Compute x*log(y) so that the result is 0 if x = 0.

Compute x*log1p(y) so that the result is 0 if x = 0.

logsumexp(a[, axis, b, keepdims, return_sign])

Compute the log of the sum of exponentials of input elements.

Relative error exponential, (exp(x) - 1)/x.

Return the normalized sinc function.

---

## gaussian_laplace#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html

**Contents:**
- gaussian_laplace#

Multidimensional Laplace filter using Gaussian second derivatives.

The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

The axes over which to apply the filter. If sigma or mode tuples are provided, their length must match the number of axes.

Filtered array. Has the same shape as input.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> ascent = datasets.ascent()
```

Example 2 (unknown):
```unknown
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
```

Example 3 (unknown):
```unknown
>>> result = ndimage.gaussian_laplace(ascent, sigma=1)
>>> ax1.imshow(result)
```

Example 4 (unknown):
```unknown
>>> result = ndimage.gaussian_laplace(ascent, sigma=3)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## bisplrep#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplrep.html

**Contents:**
- bisplrep#

Find a bivariate B-spline representation of a surface.

Given a set of data points (x[i], y[i], z[i]) representing a surface z=f(x,y), compute a B-spline representation of the surface. Based on the routine SURFIT from FITPACK.

Rank-1 arrays of data points.

Rank-1 array of weights. By default w=np.ones(len(x)).

End points of approximation interval in x. By default xb = x.min(), xe=x.max().

End points of approximation interval in y. By default yb=y.min(), ye = y.max().

The degrees of the spline (1 <= kx, ky <= 5). Third order (kx=ky=3) is recommended.

If task=0, find knots in x and y and coefficients for a given smoothing factor, s. If task=1, find knots and coefficients for another value of the smoothing factor, s. bisplrep must have been previously called with task=0 or task=1. If task=-1, find coefficients for a given set of knots tx, ty.

A non-negative smoothing factor. If weights correspond to the inverse of the standard-deviation of the errors in z, then a good s-value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m=len(x).

A threshold for determining the effective rank of an over-determined linear system of equations (0 < eps < 1). eps is not likely to need changing.

Rank-1 arrays of the knots of the spline for task=-1

Non-zero to return optional outputs.

Over-estimates of the total number of knots. If None then nxest = max(kx+sqrt(m/2),2*kx+3), nyest = max(ky+sqrt(m/2),2*ky+3).

Non-zero to suppress printing of messages.

A list [tx, ty, c, kx, ky] containing the knots (tx, ty) and coefficients (c) of the bivariate B-spline representation of the surface along with the degree of the spline.

The weighted sum of squared residuals of the spline approximation.

An integer flag about splrep success. Success is indicated if ier<=0. If ier in [1,2,3] an error occurred but was not raised. Otherwise an error is raised.

A message corresponding to the integer flag, ier.

See bisplev to evaluate the value of the B-spline given its tck representation.

If the input data is such that input dimensions have incommensurate units and differ by many orders of magnitude, the interpolant may have numerical artifacts. Consider rescaling the data before interpolation.

Dierckx P.:An algorithm for surface fitting with spline functions Ima J. Numer. Anal. 1 (1981) 267-283.

Dierckx P.:An algorithm for surface fitting with spline functions report tw50, Dept. Computer Science,K.U.Leuven, 1980.

Dierckx P.:Curve and surface fitting with splines, Monographs on Numerical Analysis, Oxford University Press, 1993.

Examples are given in the tutorial.

---

## gaussian_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

**Contents:**
- gaussian_filter#

Multidimensional Gaussian filter.

Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.

The order of the filter along each axis is given as a sequence of integers, or as a single number. An order of 0 corresponds to convolution with a Gaussian kernel. A positive order corresponds to convolution with that derivative of a Gaussian.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Truncate the filter at this many standard deviations. Default is 4.0.

Radius of the Gaussian kernel. The radius are given for each axis as a sequence, or as a single number, in which case it is equal for all axes. If specified, the size of the kernel along each axis will be 2*radius + 1, and truncate is ignored. Default is None.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for sigma, order, mode and/or radius must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

Returned array of same shape as input.

The multidimensional filter is implemented as a sequence of 1-D convolution filters. The intermediate arrays are stored in the same data type as the output. Therefore, for output types with a limited precision, the results may be imprecise because intermediate results may be stored with insufficient precision.

The Gaussian kernel will have size 2*radius + 1 along each axis. If radius is None, the default radius = round(truncate * sigma) will be used.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.ndimage import gaussian_filter
>>> import numpy as np
>>> a = np.arange(50, step=2).reshape((5,5))
>>> a
array([[ 0,  2,  4,  6,  8],
       [10, 12, 14, 16, 18],
       [20, 22, 24, 26, 28],
       [30, 32, 34, 36, 38],
       [40, 42, 44, 46, 48]])
>>> gaussian_filter(a, sigma=1)
array([[ 4,  6,  8,  9, 11],
       [10, 12, 14, 15, 17],
       [20, 22, 24, 25, 27],
       [29, 31, 33, 34, 36],
       [35, 37, 39, 40, 42]])
```

Example 2 (python):
```python
>>> from scipy import datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = gaussian_filter(ascent, sigma=5)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## irfft2#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfft2.html

**Contents:**
- irfft2#

Computes the inverse of rfft2

Shape of the real output to the inverse FFT.

The axes over which to compute the inverse fft. Default is the last two axes.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The result of the inverse real 2-D FFT.

The 2-D FFT of real input.

The inverse of the 1-D FFT of real input.

The inverse of the N-D FFT of real input.

This is really irfftn with different defaults. For more details see irfftn.

---

## LowLevelCallable#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html

**Contents:**
- LowLevelCallable#

Low-level callback function.

Some functions in SciPy take as arguments callback functions, which can either be python callables or low-level compiled functions. Using compiled callback functions can improve performance somewhat by avoiding wrapping data in Python objects.

Such low-level functions in SciPy are wrapped in LowLevelCallable objects, which can be constructed from function pointers obtained from ctypes, cffi, Cython, or contained in Python PyCapsule objects.

Functions accepting low-level callables:

scipy.integrate.quad, scipy.ndimage.generic_filter, scipy.ndimage.generic_filter1d, scipy.ndimage.geometric_transform

Extending scipy.ndimage in C, Faster integration using low-level callback functions

Low-level callback function.

User data to pass on to the callback function.

Signature of the function. If omitted, determined from function, if possible.

Callback function given.

Signature of the function.

from_cython(module, name[, user_data, signature])

Create a low-level callback function from an exported Cython function.

The argument function can be one of:

PyCapsule, whose name contains the C function signature

ctypes function pointer

cffi function pointer

The signature of the low-level callback must match one of those expected by the routine it is passed to.

If constructing low-level functions from a PyCapsule, the name of the capsule must be the corresponding signature, in the format:

The context of a PyCapsule passed in as function is used as user_data, if an explicit value for user_data was not given.

**Examples:**

Example 1 (unknown):
```unknown
return_type (arg1_type, arg2_type, ...)
```

Example 2 (unknown):
```unknown
"void (double)"
"double (double, int *, void *)"
```

---

## Spatial Data Structures and Algorithms (scipy.spatial)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/spatial.html

**Contents:**
- Spatial Data Structures and Algorithms (scipy.spatial)#
- Delaunay triangulations#
  - Coplanar points#
- Convex hulls#
- Voronoi diagrams#

scipy.spatial can compute triangulations, Voronoi diagrams, and convex hulls of a set of points, by leveraging the Qhull library.

Moreover, it contains KDTree implementations for nearest-neighbor point queries, and utilities for distance computations in various metrics.

The Delaunay triangulation is a subdivision of a set of points into a non-overlapping set of triangles, such that no point is inside the circumcircle of any triangle. In practice, such triangulations tend to avoid triangles with small angles.

Delaunay triangulation can be computed using scipy.spatial as follows:

And add some further decorations:

The structure of the triangulation is encoded in the following way: the simplices attribute contains the indices of the points in the points array that make up the triangle. For instance:

Moreover, neighboring triangles can also be found:

What this tells us is that this triangle has triangle #0 as a neighbor, but no other neighbors. Moreover, it tells us that neighbor 0 is opposite the vertex 1 of the triangle:

Indeed, from the figure, we see that this is the case.

Qhull can also perform tessellations to simplices for higher-dimensional point sets (for instance, subdivision into tetrahedra in 3-D).

It is important to note that not all points necessarily appear as vertices of the triangulation, due to numerical precision issues in forming the triangulation. Consider the above with a duplicated point:

Observe that point #4, which is a duplicate, does not occur as a vertex of the triangulation. That this happened is recorded:

This means that point 4 resides near triangle 0 and vertex 3, but is not included in the triangulation.

Note that such degeneracies can occur not only because of duplicated points, but also for more complicated geometrical reasons, even in point sets that at first sight seem well-behaved.

However, Qhull has the “QJ” option, which instructs it to perturb the input data randomly until degeneracies are resolved:

Two new triangles appeared. However, we see that they are degenerate and have zero area.

A convex hull is the smallest convex object containing all points in a given point set.

These can be computed via the Qhull wrappers in scipy.spatial as follows:

The convex hull is represented as a set of N 1-D simplices, which in 2-D means line segments. The storage scheme is exactly the same as for the simplices in the Delaunay triangulation discussed above.

We can illustrate the above result:

The same can be achieved with scipy.spatial.convex_hull_plot_2d.

A Voronoi diagram is a subdivision of the space into the nearest neighborhoods of a given set of points.

There are two ways to approach this object using scipy.spatial. First, one can use the KDTree to answer the question “which of the points is closest to this one”, and define the regions that way:

So the point (0.1, 0.1) belongs to region 0. In color:

This does not, however, give the Voronoi diagram as a geometrical object.

The representation in terms of lines and points can be again obtained via the Qhull wrappers in scipy.spatial:

The Voronoi vertices denote the set of points forming the polygonal edges of the Voronoi regions. In this case, there are 9 different regions:

Negative value -1 again indicates a point at infinity. Indeed, only one of the regions, [0, 1, 3, 2], is bounded. Note here that due to similar numerical precision issues as in Delaunay triangulation above, there may be fewer Voronoi regions than input points.

The ridges (lines in 2-D) separating the regions are described as a similar collection of simplices as the convex hull pieces:

These numbers present the indices of the Voronoi vertices making up the line segments. -1 is again a point at infinity — only 4 of the 12 lines are a bounded line segment, while others extend to infinity.

The Voronoi ridges are perpendicular to the lines drawn between the input points. To which two points each ridge corresponds is also recorded:

This information, taken together, is enough to construct the full diagram.

We can plot it as follows. First, the points and the Voronoi vertices:

Plotting the finite line segments goes as for the convex hull, but now we have to guard for the infinite edges:

The ridges extending to infinity require a bit more care:

This plot can also be created using scipy.spatial.voronoi_plot_2d.

Voronoi diagrams can be used to create interesting generative art. Try playing with the settings of this mandala function to create your own!

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.spatial import Delaunay
>>> import numpy as np
>>> points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
>>> tri = Delaunay(points)
```

Example 2 (typescript):
```typescript
>>> import matplotlib.pyplot as plt
>>> plt.triplot(points[:,0], points[:,1], tri.simplices)
>>> plt.plot(points[:,0], points[:,1], 'o')
```

Example 3 (unknown):
```unknown
>>> for j, p in enumerate(points):
...     plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points
>>> for j, s in enumerate(tri.simplices):
...     p = points[s].mean(axis=0)
...     plt.text(p[0], p[1], '#%d' % j, ha='center') # label triangles
>>> plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
>>> plt.show()
```

Example 4 (json):
```json
>>> i = 1
>>> tri.simplices[i,:]
array([3, 1, 0], dtype=int32)
>>> points[tri.simplices[i,:]]
array([[ 1. ,  1. ],
       [ 0. ,  1.1],
       [ 0. ,  0. ]])
```

---

## gaussian_filter1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html

**Contents:**
- gaussian_filter1d#

standard deviation for Gaussian kernel

The axis of input along which to calculate. Default is -1.

An order of 0 corresponds to convolution with a Gaussian kernel. A positive order corresponds to convolution with that derivative of a Gaussian.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Truncate the filter at this many standard deviations. Default is 4.0.

Radius of the Gaussian kernel. If specified, the size of the kernel will be 2*radius + 1, and truncate is ignored. Default is None.

The Gaussian kernel will have size 2*radius + 1 along each axis. If radius is None, a default radius = round(truncate * sigma) will be used.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.ndimage import gaussian_filter1d
>>> import numpy as np
>>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
>>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> x = rng.standard_normal(101).cumsum()
>>> y3 = gaussian_filter1d(x, 3)
>>> y6 = gaussian_filter1d(x, 6)
>>> plt.plot(x, 'k', label='original data')
>>> plt.plot(y3, '--', label='filtered, sigma=3')
>>> plt.plot(y6, ':', label='filtered, sigma=6')
>>> plt.legend()
>>> plt.grid()
>>> plt.show()
```

---

## Smoothing splines#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html

**Contents:**
- Smoothing splines#
- Spline smoothing in 1D#
  - “Classic” smoothing splines and generalized cross-validation (GCV) criterion#
    - Batching of y arrays#
  - Smoothing splines with automatic knot selection#
  - Smoothing spline curves in \(d>1\)#
  - Batching of y arrays#
  - Legacy routines for spline smoothing in 1-D#
    - Procedural (splrep)#
    - Manipulating spline objects: procedural (splXXX)#

For the interpolation problem, the task is to construct a curve which passes through a given set of data points. This may be not appropriate if the data is noisy: we then want to construct a smooth curve, \(g(x)\), which approximates input data without passing through each point exactly.

To this end, scipy.interpolate allows constructing smoothing splines which balance how close the resulting curve, \(g(x)\), is to the data, and the smoothness of \(g(x)\). Mathematically, the task is to solve a penalized least-squares problem, where the penalty controls the smoothness of \(g(x)\).

We provide two approaches to constructing smoothing splines, which differ in (1) the form of the penalty term, and (2) the basis in which the smoothing curve is constructed. Below we consider these two approaches.

The former variant is performed by the make_smoothing_spline function, which is a clean-room reimplementation of the classic gcvspl Fortran package by H. Woltring. The latter variant is implemented by the make_splrep function, which is a reimplementation of the Fortran FITPACK library by P. Dierckx. A legacy interface to the FITPACK library is also available.

Given the data arrays x and y and the array of non-negative weights, w, we look for a cubic spline function g(x) which minimizes

where \(\lambda \geqslant 0\) is a non-negative penalty parameter, and \(g^{(2)}(x)\) is the second derivative of \(g(x)\). The summation in the first term runs over the data points, \((x_j, y_j)\), and the integral in the second term is over the whole interval \(x \in [x_1, x_n]\).

Here the first term penalizes the deviation of the spline function from the data, and the second term penalizes large values of the second derivative—which is taken as the criterion for the smoothness of a curve.

The target function, \(g(x)\), is taken to be a natural cubic spline with knots at the data points, \(x_j\), and the minimization is carried over the spline coefficients at a given value of \(\lambda\).

Clearly, \(\lambda = 0\) corresponds to the interpolation problem—the result is a natural interpolating spline; in the opposite limit, \(\lambda \gg 1\), the result \(g(x)\) approaches a straight line (since the minimization effectively zeros out the second derivative of \(g(x)\)).

The smoothing function strongly depends on \(\lambda\), and multiple strategies are possible for selecting an “optimal” value of the penalty. One popular strategy is the so-called generalized cross-validation (GCV): conceptually, this is equivalent to comparing the spline functions constructed on reduced sets of data where we leave out one data point. Direct application of this leave-one-out cross-validation procedure is very costly, and we use a more efficient GCV algorithm.

To construct the smoothing spline given data and the penalty parameter, we use the function make_smoothing_spline. Its interface is similar to the constructor of interpolating splines, make_interp_spline: it accepts data arrays and returns a callable BSpline instance.

Additionally, it accepts an optional lam keyword argument to specify the penalty parameter \(\lambda\). If omitted or set to None, \(\lambda\) is computed via the GCV procedure.

To illustrate the effect of the penalty parameter, consider a toy example of a sine curve with some noise:

Generate some noisy data:

Construct and plot smoothing splines for a series of values of the penalty parameter:

We clearly see that lam=0 constructs the interpolating spline; large values of lam flatten out the resulting curve towards a straight line; and the GCV result, lam=None, is close to the underlying sine curve.

make_smoothing_spline constructor accepts multidimensional y arrays and an optional axis parameter and interprets them exactly the same way the interpolating spline constructor, make_interp_spline does. See the interpolation section for a discussion and examples.

As an addition to make_smoothing_spline, SciPy provides an alternative, in the form of make_splrep and make_splprep routines. The former constructs spline functions and the latter is for parametric spline curves in \(d > 1\) dimensions.

While having a similar API (receive the data arrays, return a BSpline instance), these differ from make_smoothing_spline in several ways:

the functional form of the penalty term is different: these routines use jumps of the \(k\)-th derivative instead of the integral of the \((k-1)\)-th derivative;

instead of the penalty parameter \(\lambda\), a smoothness parameter \(s\) is used;

these routines automatically construct the knot vector; depending on inputs, resulting splines may have much fewer knots than data points.

by default boundary conditions differ: while make_smoothing_spline constructs natural cubic splines, these routines use the not-a-knot boundary conditions by default.

Let us consider the algorithm in more detail. First, the smoothing criterion. Given a cubic spline function, \(g(x)\), defined by the knots, \(t_j\), and coefficients, \(c_j\), consider the jumps of the third derivative at internal knots,

(For degree-\(k\) splines, we would have used jumps of the \(k\)-th derivative.)

If all \(D_j = 0\), then \(g(x)\) is a single polynomial on the whole domain spanned by the knots. We thus consider \(g(x)\) to be a piecewise \(C^2\)-differentiable spline function and use as the smoothing criterion the sum of jumps,

where the minimization performed is over the spline coefficients, and, potentially, the spline knots.

To make sure \(g(x)\) approximates the input data, \(x_j\) and \(y_j\), we introduce the smoothness parameter \(s \geqslant 0\) and add a constraint that

In this formulation, the smoothness parameter \(s\) is a user input, much like the penalty parameter \(\lambda\) is for the classic smoothing splines.

Note that the limit s = 0 corresponds to the interpolation problem where \(g(x_j) = y_j\). Increasing s leads to smoother fits, and in the limit of a very large s, \(g(x)\) degenerates into a single best-fit polynomial.

For a fixed knot vector and a given value of \(s\), the minimization problem is linear. If we also minimize with respect to the knots, the problem becomes non-linear. We thus need to specify an iterative minimization process to construct the knot vector along with the spline coefficients.

We therefore use the following procedure:

we start with a spline with no internal knots, and check the smoothness condition for the user-provided value of \(s\). If it is satisfied, we are done. Otherwise,

iterate, where on each iteration we

add new knots by splitting the interval with the maximum deviation between the spline function \(g(x_j)\) and the data \(y_j\).

construct the next approximation for \(g(x)\) and check the smoothness criterion.

The iterations stop if either the smoothness condition is satisfied, or the maximum allowed number of knots is reached. The latter can be either specified by a user, or is taken as the default value len(x) + k + 1 which corresponds to the interpolation of the data array x with splines of degree k.

Rephrasing and glossing over details, the procedure is to iterate over the knot vectors generated by generate_knots, applying make_lsq_spline on each step. In pseudocode:

For s=0, we take a short-cut and construct the interpolating spline with the not-a-knot boundary condition instead of iterating.

The iterative procedure of constructing a knot vector is available through the generator function generate_knots. To illustrate:

For s=0, the generator cuts short:

In general, knots are placed at data sites. The exception is even-order splines, where the knots can be placed away from data. This happens when s=0 (interpolation), or when s is small enough so that the maximum number of knots is reached, and the routine switches to the s=0 knot vector (sometimes known as Greville abscissae).

The heuristics for constructing the knot vector follows the algorithm used by the FITPACK Fortran library. The algorithm is the same, and small differences are possible due to floating-point rounding.

We now illustrate make_splrep results, using the same toy dataset as in the previous section

Generate some noisy data

Construct and plot smoothing splines for a series of values of the s parameter:

We see that the \(s=0\) curve follows the (random) fluctuations of the data points, while the \(s > 0\) curve is close to the underlying sine function. Also note that the extrapolated values vary wildly depending on the value of \(s\).

Finding a good value of \(s\) is a trial-and-error process. If the weights correspond to the inverse of standard deviations of the input data, a “good” value of \(s\) is expected to be somewhere between \(m - \sqrt{2m}\) and \(m + \sqrt{2m}\), where \(m\) is the number of data points. If all weights equal unity, a reasonable choice might be around \(s \sim m\,\sigma^2\), where \(\sigma\) is an estimate for the standard deviation of the data.

The number of knots a very strongly dependent on s. It is possible that small variations of s lead to drastic changes in the knot number.

Both make_smoothing_spline and make_splrep allow for weighted fits, where the user provides an array of weights, w. Note that the definition differs somewhat: make_smoothing_spline squares the weights to be consistent with gcvspl, while make_splrep does not—to be consistent with FITPACK.

So far we considered constructing smoothing spline functions, \(g(x)\) given data arrays x and y. We now consider a related problem of constructing a smoothing spline curve, where we consider the data as points on a plane, \(\mathbf{p}_j = (x_j, y_j)\), and we want to construct a parametric function \(\mathbf{g}(\mathbf{p}) = (g_x(u), g_y(u))\), where the values of the parameter \(u_j\) correspond to \(x_j\) and \(y_j\).

Note that this problem readily generalizes to higher dimensions, \(d > 2\): we simply have \(d\) data arrays and construct a parametric function with \(d\) components.

Also note that the choice of parametrization cannot be automated, and different parameterizations can lead to very different curves for the same data, even for interpolating curves.

Once a specific form of parametrization is chosen, the problem of constructing a smoothing curve is conceptually very similar to constructing a smoothing spline function. In a nutshell,

spline knots are added from the values of the parameter \(u\), and

both the cost function to minimize and the constraint we considered for spline functions simply get an extra summation over the \(d\) components.

The “parametric” generalization of the make_splrep function is make_splprep, and its docstring spells out the precise mathematical formulation of the minimization problem it solves.

The main user-visible difference of the parametric case is the user interface:

instead of two data arrays, x and y, make_splprep receives a single two-dimensional array, where the second dimension has size \(d\) and each data array is stored along the first dimension (alternatively, you can supply a list of 1D arrays).

the return value is pair: a BSpline instance and the array of parameter values, u, which corresponds to the input data arrays.

By default, make_splprep constructs and returns the cord length parametrization of input data (see the Parametric spline curves section for details). Alternatively, you can provide your own array of parameter values, u.

To illustrate the API, consider a toy problem: we have some data sampled from a folium of Descartes plus some noise.

Add some noise and construct the interpolators

And plot the results (the result of spl(u) is a 2D array, so we unpack it into a pair of x and y arrays for plotting).

Unlike interpolating splines and GCV smoothers, make_splrep and make_splprep do not allow multidimensional y arrays and require that x.ndim == y.ndim == 1.

The technical reason for this limitation is that the length of the knot vector t depends on the y values, thus for a batched y, the batched t could be a ragged array, which BSpline is not equipped to handle.

Therefore if you need to handle batched inputs, you will need to loop over the batch manually and construct a BSpline object per slice of the batch. Having done that, you can mimic BSpline behavior for evaluations with a workaround along the lines of

In addition to smoothing splines constructors we discussed in the previous sections, scipy.interpolate provides direct interfaces for routines from the venerable FITPACK Fortran library authored by P. Dierckx.

These interfaces should be considered legacy—while we do not plan to deprecate or remove them, we recommend that new code uses more modern alternatives, make_smoothing_spline, make_splrep or make_splprep, instead.

For historical reasons, scipy.interpolate provides two equivalent interfaces for FITPACK, a interface and an object-oriented interface. While equivalent, these interfaces have different defaults. Below we discuss them in turn, starting from the functional interface.

Spline interpolation requires two essential steps: (1) a spline representation of the curve is computed, and (2) the spline is evaluated at the desired points. In order to find the spline representation, there are two different ways to represent a curve and obtain (smoothing) spline coefficients: directly and parametrically. The direct method finds the spline representation of a curve in a 2-D plane using the function splrep. The first two arguments are the only ones required, and these provide the \(x\) and \(y\) components of the curve. The normal output is a 3-tuple, \(\left(t,c,k\right)\) , containing the knot-points, \(t\) , the coefficients \(c\) and the order \(k\) of the spline. The default spline order is cubic, but this can be changed with the input keyword, k.

The knot array defines the interpolation interval to be t[k:-k], so that the first \(k+1\) and last \(k+1\) entries of the t array define boundary knots. The coefficients are a 1D array of length at least len(t) - k - 1. Some routines pad this array to have len(c) == len(t)— these additional coefficients are ignored for the spline evaluation.

The tck-tuple format is compatible with interpolating b-splines: the output of splrep can be wrapped into a BSpline object, e.g. BSpline(*tck), and the evaluation/integration/root-finding routines described below can use tck-tuples and BSpline objects interchangeably.

For curves in N-D space the function splprep allows defining the curve parametrically. For this function only 1 input argument is required. This input is a list of \(N\) arrays representing the curve in N-D space. The length of each array is the number of curve points, and each array provides one component of the N-D data point. The parameter variable is given with the keyword argument, u, which defaults to an equally-spaced monotonic sequence between \(0\) and \(1\) (i.e., the uniform parametrization).

The output consists of two objects: a 3-tuple, \(\left(t,c,k\right)\) , containing the spline representation and the parameter variable \(u.\)

The coefficients are a list of \(N\) arrays, where each array corresponds to a dimension of the input data. Note that the knots, t correspond to the parametrization of the curve u.

The keyword argument, s , is used to specify the amount of smoothing to perform during the spline fit. The default value of \(s\) is \(s=m-\sqrt{2m}\) where \(m\) is the number of data points being fit. Therefore, if no smoothing is desired a value of \(\mathbf{s}=0\) should be passed to the routines.

Once the spline representation of the data has been determined, it can be evaluated either using the splev function or by wrapping the tck tuple into a BSpline object, as demonstrated below.

We start by illustrating the effect of the s parameter on smoothing some synthetic noisy data

Generate some noisy data

Construct two splines with different values of s.

We see that the s=0 curve follows the (random) fluctuations of the data points, while the s > 0 curve is close to the underlying sine function. Also note that the extrapolated values vary wildly depending on the value of s.

The default value of s depends on whether the weights are supplied or not, and also differs for splrep and splprep. Therefore, we recommend always providing the value of s explicitly.

Once the spline representation of the data has been determined, functions are available for evaluating the spline (splev) and its derivatives (splev, spalde) at any point and the integral of the spline between any two points ( splint). In addition, for cubic splines ( \(k=3\) ) with 8 or more knots, the roots of the spline can be estimated ( sproot). These functions are demonstrated in the example that follows.

Note that the last line is equivalent to BSpline(*tck)(xnew).

All derivatives of spline

Notice that sproot may fail to find an obvious solution at the edge of the approximation interval, \(x = 0\). If we define the spline on a slightly larger interval, we recover both roots \(x = 0\) and \(x = \pi\):

Note that in the last example, splprep returns the spline coefficients as a list of arrays, where each array corresponds to a dimension of the input data. Thus to wrap its output to a BSpline, we need to transpose the coefficients (or use BSpline(..., axis=1)):

The spline-fitting capabilities described above are also available via an objected-oriented interface. The 1-D splines are objects of the UnivariateSpline class, and are created with the \(x\) and \(y\) components of the curve provided as arguments to the constructor. The class defines __call__, allowing the object to be called with the x-axis values, at which the spline should be evaluated, returning the interpolated y-values. This is shown in the example below for the subclass InterpolatedUnivariateSpline. The integral, derivatives, and roots methods are also available on UnivariateSpline objects, allowing definite integrals, derivatives, and roots to be computed for the spline.

The UnivariateSpline class can also be used to smooth data by providing a non-zero value of the smoothing parameter s, with the same meaning as the s keyword of the splrep function described above. This results in a spline that has fewer knots than the number of data points, and hence is no longer strictly an interpolating spline, but rather a smoothing spline. If this is not desired, the InterpolatedUnivariateSpline class is available. It is a subclass of UnivariateSpline that always passes through all points (equivalent to forcing the smoothing parameter to 0). This class is demonstrated in the example below.

The LSQUnivariateSpline class is the other subclass of UnivariateSpline. It allows the user to specify the number and location of internal knots explicitly with the parameter t. This allows for the creation of customized splines with non-linear spacing, to interpolate in some domains and smooth in others, or change the character of the spline.

InterpolatedUnivariateSpline

LSQUnivarateSpline with non-uniform knots

In addition to smoothing 1-D splines, the FITPACK library provides the means of fitting 2-D surfaces to two-dimensional data. The surfaces can be thought of as functions of two arguments, \(z = g(x, y)\), constructed as tensor products of 1-D splines.

Assuming that the data is held in three arrays, x, y and z, there are two ways these data arrays can be interpreted. First—the scattered interpolation problem—the data is assumed to be paired, i.e. the pairs of values x[i] and y[i] represent the coordinates of the point i, which corresponds to z[i].

The surface \(g(x, y)\) is constructed to satisfy

where \(w_i\) are non-negative weights, and s is the input parameter, known as the smoothing factor, which controls the interplay between smoothness of the resulting function g(x, y) and the quality of the approximation of the data (i.e., the differences between \(g(x_i, y_i)\) and \(z_i\)). The limit of \(s = 0\) formally corresponds to interpolation, where the surface passes through the input data, \(g(x_i, y_i) = z_i\). See the note below however.

The second case—the rectangular grid interpolation problem—is where the data points are assumed to be on a rectangular grid defined by all pairs of the elements of the x and y arrays. For this problem, the z array is assumed to be two-dimensional, and z[i, j] corresponds to (x[i], y[j]). The bivariate spline function \(g(x, y)\) is constructed to satisfy

where s is the smoothing factor. Here the limit of \(s=0\) also formally corresponds to interpolation, \(g(x_i, y_j) = z_{i, j}\).

Internally, the smoothing surface \(g(x, y)\) is constructed by placing spline knots into the bounding box defined by the data arrays. The knots are placed automatically via the FITPACK algorithm until the desired smoothness is reached.

The knots may be placed away from the data points.

While \(s=0\) formally corresponds to a bivariate spline interpolation, the FITPACK algorithm is not meant for interpolation, and may lead to unexpected results.

For scattered data interpolation, prefer griddata; for data on a regular grid, prefer RegularGridInterpolator.

If the input data, x and y, is such that input dimensions have incommensurate units and differ by many orders of magnitude, the interpolant \(g(x, y)\) may have numerical artifacts. Consider rescaling the data before interpolation.

We now consider the two spline fitting problems in turn.

There are two interfaces for the underlying FITPACK library, a procedural one and an object-oriented interface.

Procedural interface (`bisplrep`)

For (smooth) spline fitting to a 2-D surface, the function bisplrep is available. This function takes as required inputs the 1-D arrays x, y, and z, which represent points on the surface \(z=f(x, y).\) The spline orders in x and y directions can be specified via the optional parameters kx and ky. The default is a bicubic spline, kx=ky=3.

The output of bisplrep is a list [tx ,ty, c, kx, ky] whose entries represent respectively, the components of the knot positions, the coefficients of the spline, and the order of the spline in each coordinate. It is convenient to hold this list in a single object, tck, so that it can be passed easily to the function bisplev. The keyword, s , can be used to change the amount of smoothing performed on the data while determining the appropriate spline. The recommended values for \(s\) depend on the weights \(w_i\). If these are taken as \(1/d_i\), with \(d_i\) an estimate of the standard deviation of \(z_i\), a good value of \(s\) should be found in the range \(m- \sqrt{2m}, m + \sqrt{2m}\), where where \(m\) is the number of data points in the x, y, and z vectors.

The default value is \(s=m-\sqrt{2m}\). As a result, if no smoothing is desired, then ``s=0`` should be passed to `bisplrep`. (See however the note above).

To evaluate the 2-D spline and its partial derivatives (up to the order of the spline), the function bisplev is required. This function takes as the first two arguments two 1-D arrays whose cross-product specifies the domain over which to evaluate the spline. The third argument is the tck list returned from bisplrep. If desired, the fourth and fifth arguments provide the orders of the partial derivative in the \(x\) and \(y\) direction, respectively.

It is important to note that 2-D interpolation should not be used to find the spline representation of images. The algorithm used is not amenable to large numbers of input points. scipy.signal and scipy.ndimage contain more appropriate algorithms for finding the spline representation of an image.

The 2-D interpolation commands are intended for use when interpolating a 2-D function as shown in the example that follows. This example uses the mgrid command in NumPy which is useful for defining a “mesh-grid” in many dimensions. (See also the ogrid command if the full-mesh is not needed). The number of output arguments and the number of dimensions of each argument is determined by the number of indexing objects passed in mgrid.

Define function over a sparse 20x20 grid

Interpolate function over a new 70x70 grid

Object-oriented interface (`SmoothBivariateSpline`)

The object-oriented interface for bivariate spline smoothing of scattered data, SmoothBivariateSpline class, implements a subset of the functionality of the bisplrep / bisplev pair, and has different defaults.

It takes the elements of the weights array equal unity, \(w_i = 1\) and constructs the knot vectors automatically given the input value of the smoothing factor s— the default value is \(m\), the number of data points.

The spline orders in the x and y directions are controlled by the optional parameters kx and ky, with the default of kx=ky=3.

We illustrate the effect of the smoothing factor using the following example:

Here we take a known function (displayed at the leftmost panel), sample it on a mesh of points (shown by white dots), and construct the spline fit using the default smoothing (center panel) and forcing the interpolation (rightmost panel).

Several features are clearly visible. First, the default value of s provides too much smoothing for this data; forcing the interpolation condition, s = 0, allows to restore the underlying function to a reasonable accuracy. Second, outside of the interpolation range (i.e., the area covered by white dots) the result is extrapolated using a nearest-neighbor constant. Finally, we had to silence the warnings (which is a bad form, yes!).

The warning here is emitted in the s=0 case, and signals an internal difficulty FITPACK encountered when we forced the interpolation condition. If you see this warning in your code, consider switching to bisplrep and increase its nxest, nyest parameters (see the bisplrep docstring for more details).

For gridded 2D data, fitting a smoothing tensor product spline can be done using the RectBivariateSpline class. It has the interface similar to that of SmoothBivariateSpline, the main difference is that the 1D input arrays x and y are understood as defining a 2D grid (as their outer product), and the z array is 2D with the shape of len(x) by len(y).

The spline orders in the x and y directions are controlled by the optional parameters kx and ky, with the default of kx=ky=3, i.e. a bicubic spline.

The default value of the smoothing factor is s=0. We nevertheless recommend to always specify s explicitly.

If your data is given in spherical coordinates, \(r = r(\theta, \phi)\), SmoothSphereBivariateSpline and RectSphereBivariateSpline provide convenient analogs of SmoothBivariateSpline and RectBivariateSpline, respectively.

These classes ensure the periodicity of the spline fits for \(\theta \in [0, \pi]\) and \(\phi \in [0, 2\pi]\), and offer some control over the continuity at the poles. Refer to the docstrings of these classes for details.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.interpolate import make_smoothing_spline
```

Example 2 (unknown):
```unknown
>>> x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/16)
>>> rng = np.random.default_rng()
>>> y =  np.sin(x) + 0.4*rng.standard_normal(size=len(x))
```

Example 3 (bash):
```bash
>>> import matplotlib.pyplot as plt
>>> xnew = np.arange(0, 9/4, 1/50) * np.pi
>>> for lam in [0, 0.02, 10, None]:
...     spl = make_smoothing_spline(x, y, lam=lam)
...     plt.plot(xnew, spl(xnew), '-.', label=fr'$\lambda=${lam}')
>>> plt.plot(x, y, 'o')
>>> plt.legend()
>>> plt.show()
```

Example 4 (typescript):
```typescript
for t in generate_knots(x, y, s=s):
    g = make_lsq_spline(x, y, t=t)     # construct
    if ((y - g(x))**2).sum() < s:      # check smoothness
        break
```

---

## Batched Linear Operations#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/linalg_batch.html

**Contents:**
- Batched Linear Operations#

Almost all of SciPy’s linear algebra functions now support N-dimensional array input. These operations have not been mathematically generalized to higher-order tensors; rather, the indicated operation is performed on a batch (or “stack”) of input scalars, vectors, and/or matrices.

Consider the linalg.det function, which maps a matrix to a scalar.

Sometimes we need the determinant of a batch of matrices of the same dimensionality.

We could perform the operation for each element of the batch in a loop or list comprehension:

However, just as we might use NumPy broadcasting and vectorization rules to create the batch of matrices in the first place:

we might also wish to perform the determinant operation on all of the matrices in one function call.

In SciPy, we prefer the term “batch” instead of “stack” because the idea is generalized to N-dimensional batches. Suppose the input is a 2 x 4 batch of 3 x 3 matrices.

In this case, we say that the batch shape is (2, 4), and the core shape of the input is (3, 3). The net shape of the input is the sum (concatenation) of the batch shape and core shape.

Since each 3 x 3 matrix is converted to a zero-dimensional scalar, we say that the core shape of the outuput is (). The shape of the output is the sum of the batch shape and core shape, so the result is a 2 x 4 array.

Not all linear algebra functions map to scalars. For instance, the scipy.linalg.expm function maps from a matrix to a matrix with the same shape.

In this case, the core shape of the output is (3, 3), so with a batch shape of (2, 4), we expect an output of shape (2, 4, 3, 3).

Generalization of these rules to functions with multiple inputs and outputs is straightforward. For instance, the scipy.linalg.eig function produces two outputs by default, a vector and a matrix.

In this case, the core shape of the output vector is (3,) and the core shape of the output matrix is (3, 3). The shape of each output is the batch shape plus the core shape as before.

When there is more than one input, there is no complication if the input shapes are identical.

The rules when the shapes are not identical follow logically. Each input can have its own batch shape as long as the shapes are broadcastable according to NumPy’s broadcasting rules. The net batch shape is the broadcasted shape of the individual batch shapes, and the shape of each output is the net batch shape plus its core shape.

There are a few functions for which the core dimensionality (i.e., the length of the core shape) of an argument or output can be either 1 or 2. In these cases, the core dimensionality is taken to be 1 if the array has only one dimension and 2 if the array has two or more dimensions. For instance, consider the following calls to scipy.linalg.solve. The simplest case is a single square matrix A and a single vector b:

In this case, the core dimensionality of A is 2 (shape (5, 5)), the core dimensionality of b is 1 (shape (5,)), and the core dimensionality of the output is 1 (shape (5,)).

However, b can also be a two-dimensional array in which the columns are taken to be one-dimensional vectors.

At first glance, it might seem that the core shape of b is still (5,), and we have simply performed the operation with a batch shape of (2,). However, if this were the case, the batch shape of b would be prepended to the core shape, resulting in b and the output having shape (2, 5). Thinking more carefully, it is correct to consider the core dimensionality of both inputs and the output to be 2; the batch shape is ().

Likewise, whenever b has more than two dimensions, the core dimensionality of b and the output is considered to be 2. For example, to solve a batch of three entirely separate linear systems, each with only one right hand side, b must be provided as a three-dimensional array: one dimensions for the batch shape ((3,)) and two for the core shape ((5, 1)).

**Examples:**

Example 1 (python):
```python
import numpy as np
from scipy import linalg
A = np.eye(3)
linalg.det(A)
```

Example 2 (unknown):
```unknown
np.float64(1.0)
```

Example 3 (bash):
```bash
batch = [i*np.eye(3) for i in range(1, 4)]
batch
```

Example 4 (json):
```json
[array([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]]),
 array([[2., 0., 0.],
        [0., 2., 0.],
        [0., 0., 2.]]),
 array([[3., 0., 0.],
        [0., 3., 0.],
        [0., 0., 3.]])]
```

---

## solve_ivp#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

**Contents:**
- solve_ivp#

Solve an initial value problem for a system of ODEs.

This function numerically integrates a system of ordinary differential equations given an initial value:

Here t is a 1-D independent variable (time), y(t) is an N-D vector-valued function (state), and an N-D vector-valued function f(t, y) determines the differential equations. The goal is to find y(t) approximately satisfying the differential equations, given an initial value y(t0)=y0.

Some of the solvers support integration in the complex domain, but note that for stiff ODE solvers, the right-hand side must be complex-differentiable (satisfy Cauchy-Riemann equations [11]). To solve a problem in the complex domain, pass y0 with a complex data type. Another option always available is to rewrite your problem for real and imaginary parts separately.

Right-hand side of the system: the time derivative of the state y at time t. The calling signature is fun(t, y), where t is a scalar and y is an ndarray with len(y) = len(y0). Additional arguments need to be passed if args is used (see documentation of args argument). fun must return an array of the same shape as y. See vectorized for more information.

Interval of integration (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf. Both t0 and tf must be floats or values interpretable by the float conversion function.

Initial state. For problems in the complex domain, pass y0 with a complex data type (even if the initial value is purely real).

Integration method to use:

‘RK45’ (default): Explicit Runge-Kutta method of order 5(4) [1]. The error is controlled assuming accuracy of the fourth-order method, but steps are taken using the fifth-order accurate formula (local extrapolation is done). A quartic interpolation polynomial is used for the dense output [2]. Can be applied in the complex domain.

‘RK23’: Explicit Runge-Kutta method of order 3(2) [3]. The error is controlled assuming accuracy of the second-order method, but steps are taken using the third-order accurate formula (local extrapolation is done). A cubic Hermite polynomial is used for the dense output. Can be applied in the complex domain.

‘DOP853’: Explicit Runge-Kutta method of order 8 [13]. Python implementation of the “DOP853” algorithm originally written in Fortran [14]. A 7-th order interpolation polynomial accurate to 7-th order is used for the dense output. Can be applied in the complex domain.

‘Radau’: Implicit Runge-Kutta method of the Radau IIA family of order 5 [4]. The error is controlled with a third-order accurate embedded formula. A cubic polynomial which satisfies the collocation conditions is used for the dense output.

‘BDF’: Implicit multi-step variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation [5]. The implementation follows the one described in [6]. A quasi-constant step scheme is used and accuracy is enhanced using the NDF modification. Can be applied in the complex domain.

‘LSODA’: Adams/BDF method with automatic stiffness detection and switching [7], [8]. This is a wrapper of the Fortran solver from ODEPACK.

Explicit Runge-Kutta methods (‘RK23’, ‘RK45’, ‘DOP853’) should be used for non-stiff problems and implicit methods (‘Radau’, ‘BDF’) for stiff problems [9]. Among Runge-Kutta methods, ‘DOP853’ is recommended for solving with high precision (low values of rtol and atol).

If not sure, first try to run ‘RK45’. If it makes unusually many iterations, diverges, or fails, your problem is likely to be stiff and you should use ‘Radau’ or ‘BDF’. ‘LSODA’ can also be a good universal choice, but it might be somewhat less convenient to work with as it wraps old Fortran code.

You can also pass an arbitrary class derived from OdeSolver which implements the solver.

Times at which to store the computed solution, must be sorted and lie within t_span. If None (default), use points selected by the solver.

Whether to compute a continuous solution. Default is False.

Events to track. If None (default), no events will be tracked. Each event occurs at the zeros of a continuous function of time and state. Each function must have the signature event(t, y) where additional argument have to be passed if args is used (see documentation of args argument). Each function must return a float. The solver will find an accurate value of t at which event(t, y(t)) = 0 using a root-finding algorithm. By default, all zeros will be found. The solver looks for a sign change over each step, so if multiple zero crossings occur within one step, events may be missed. Additionally each event function might have the following attributes:

When boolean, whether to terminate integration if this event occurs. When integral, termination occurs after the specified the number of occurrences of this event. Implicitly False if not assigned.

Direction of a zero crossing. If direction is positive, event will only trigger when going from negative to positive, and vice versa if direction is negative. If 0, then either direction will trigger event. Implicitly 0 if not assigned.

You can assign attributes like event.terminal = True to any function in Python.

Whether fun can be called in a vectorized fashion. Default is False.

If vectorized is False, fun will always be called with y of shape (n,), where n = len(y0).

If vectorized is True, fun may be called with y of shape (n, k), where k is an integer. In this case, fun must behave such that fun(t, y)[:, i] == fun(t, y[:, i]) (i.e. each column of the returned array is the time derivative of the state corresponding with a column of y).

Setting vectorized=True allows for faster finite difference approximation of the Jacobian by methods ‘Radau’ and ‘BDF’, but will result in slower execution for other methods and for ‘Radau’ and ‘BDF’ in some circumstances (e.g. small len(y0)).

Additional arguments to pass to the user-defined functions. If given, the additional arguments are passed to all user-defined functions. So if, for example, fun has the signature fun(t, y, a, b, c), then jac (if given) and any event functions must have the same signature, and args must be a tuple of length 3.

Options passed to a chosen solver. All options available for already implemented solvers are listed below.

Initial step size. Default is None which means that the algorithm should choose.

Maximum allowed step size. Default is np.inf, i.e., the step size is not bounded and determined solely by the solver.

Relative and absolute tolerances. The solver keeps the local error estimates less than atol + rtol * abs(y). Here rtol controls a relative accuracy (number of correct digits), while atol controls absolute accuracy (number of correct decimal places). To achieve the desired rtol, set atol to be smaller than the smallest value that can be expected from rtol * abs(y) so that rtol dominates the allowable error. If atol is larger than rtol * abs(y) the number of correct digits is not guaranteed. Conversely, to achieve the desired atol set rtol such that rtol * abs(y) is always smaller than atol. If components of y have different scales, it might be beneficial to set different atol values for different components by passing array_like with shape (n,) for atol. Default values are 1e-3 for rtol and 1e-6 for atol.

Jacobian matrix of the right-hand side of the system with respect to y, required by the ‘Radau’, ‘BDF’ and ‘LSODA’ method. The Jacobian matrix has shape (n, n) and its element (i, j) is equal to d f_i / d y_j. There are three ways to define the Jacobian:

If array_like or sparse_matrix, the Jacobian is assumed to be constant. Not supported by ‘LSODA’.

If callable, the Jacobian is assumed to depend on both t and y; it will be called as jac(t, y), as necessary. Additional arguments have to be passed if args is used (see documentation of args argument). For ‘Radau’ and ‘BDF’ methods, the return value might be a sparse matrix.

If None (default), the Jacobian will be approximated by finite differences.

It is generally recommended to provide the Jacobian rather than relying on a finite-difference approximation.

Defines a sparsity structure of the Jacobian matrix for a finite- difference approximation. Its shape must be (n, n). This argument is ignored if jac is not None. If the Jacobian has only few non-zero elements in each row, providing the sparsity structure will greatly speed up the computations [10]. A zero entry means that a corresponding element in the Jacobian is always zero. If None (default), the Jacobian is assumed to be dense. Not supported by ‘LSODA’, see lband and uband instead.

Parameters defining the bandwidth of the Jacobian for the ‘LSODA’ method, i.e., jac[i, j] != 0 only for i - lband <= j <= i + uband. Default is None. Setting these requires your jac routine to return the Jacobian in the packed format: the returned array must have n columns and uband + lband + 1 rows in which Jacobian diagonals are written. Specifically jac_packed[uband + i - j , j] = jac[i, j]. The same format is used in scipy.linalg.solve_banded (check for an illustration). These parameters can be also used with jac=None to reduce the number of Jacobian elements estimated by finite differences.

The minimum allowed step size for ‘LSODA’ method. By default min_step is zero.

Values of the solution at t.

Found solution as OdeSolution instance; None if dense_output was set to False.

Contains for each event type a list of arrays at which an event of that type event was detected. None if events was None.

For each value of t_events, the corresponding value of the solution. None if events was None.

Number of evaluations of the right-hand side.

Number of evaluations of the Jacobian.

Number of LU decompositions.

Reason for algorithm termination:

-1: Integration step failed.

0: The solver successfully reached the end of tspan.

1: A termination event occurred.

Human-readable description of the termination reason.

True if the solver reached the interval end or a termination event occurred (status >= 0).

J. R. Dormand, P. J. Prince, “A family of embedded Runge-Kutta formulae”, Journal of Computational and Applied Mathematics, Vol. 6, No. 1, pp. 19-26, 1980.

L. W. Shampine, “Some Practical Runge-Kutta Formulas”, Mathematics of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.

P. Bogacki, L.F. Shampine, “A 3(2) Pair of Runge-Kutta Formulas”, Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.

E. Hairer, G. Wanner, “Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems”, Sec. IV.8.

Backward Differentiation Formula on Wikipedia.

L. F. Shampine, M. W. Reichelt, “THE MATLAB ODE SUITE”, SIAM J. SCI. COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.

A. C. Hindmarsh, “ODEPACK, A Systematized Collection of ODE Solvers,” IMACS Transactions on Scientific Computation, Vol 1., pp. 55-64, 1983.

L. Petzold, “Automatic selection of methods for solving stiff and nonstiff systems of ordinary differential equations”, SIAM Journal on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148, 1983.

Stiff equation on Wikipedia.

A. Curtis, M. J. D. Powell, and J. Reid, “On the estimation of sparse Jacobian matrices”, Journal of the Institute of Mathematics and its Applications, 13, pp. 117-120, 1974.

Cauchy-Riemann equations on Wikipedia.

Lotka-Volterra equations on Wikipedia.

E. Hairer, S. P. Norsett G. Wanner, “Solving Ordinary Differential Equations I: Nonstiff Problems”, Sec. II.

Page with original Fortran code of DOP853.

Basic exponential decay showing automatically chosen time points.

Specifying points where the solution is desired.

Cannon fired upward with terminal event upon impact. The terminal and direction fields of an event are applied by monkey patching a function. Here y[0] is position and y[1] is velocity. The projectile starts at position 0 with velocity +10. Note that the integration never reaches t=100 because the event is terminal.

Use dense_output and events to find position, which is 100, at the apex of the cannonball’s trajectory. Apex is not defined as terminal, so both apex and hit_ground are found. There is no information at t=20, so the sol attribute is used to evaluate the solution. The sol attribute is returned by setting dense_output=True. Alternatively, the y_events attribute can be used to access the solution at the time of the event.

As an example of a system with additional parameters, we’ll implement the Lotka-Volterra equations [12].

We pass in the parameter values a=1.5, b=1, c=3 and d=1 with the args argument.

Compute a dense solution and plot it.

A couple examples of using solve_ivp to solve the differential equation y' = Ay with complex matrix A.

Solving an IVP with A from above and y as 3x1 vector:

Solving an IVP with A from above with y as 3x3 matrix :

**Examples:**

Example 1 (unknown):
```unknown
dy / dt = f(t, y)
y(t0) = y0
```

Example 2 (python):
```python
>>> import numpy as np
>>> from scipy.integrate import solve_ivp
>>> def exponential_decay(t, y): return -0.5 * y
>>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
>>> print(sol.t)
[ 0.          0.11487653  1.26364188  3.06061781  4.81611105  6.57445806
  8.33328988 10.        ]
>>> print(sol.y)
[[2.         1.88836035 1.06327177 0.43319312 0.18017253 0.07483045
  0.03107158 0.01350781]
 [4.         3.7767207  2.12654355 0.86638624 0.36034507 0.14966091
  0.06214316 0.02701561]
 [8.         7.5534414  4.25308709 1.73277247 0.72069014 0.29932181
  0.12428631 0.05403123]]
```

Example 3 (json):
```json
>>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8],
...                 t_eval=[0, 1, 2, 4, 10])
>>> print(sol.t)
[ 0  1  2  4 10]
>>> print(sol.y)
[[2.         1.21305369 0.73534021 0.27066736 0.01350938]
 [4.         2.42610739 1.47068043 0.54133472 0.02701876]
 [8.         4.85221478 2.94136085 1.08266944 0.05403753]]
```

Example 4 (python):
```python
>>> def upward_cannon(t, y): return [y[1], -0.5]
>>> def hit_ground(t, y): return y[0]
>>> hit_ground.terminal = True
>>> hit_ground.direction = -1
>>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)
>>> print(sol.t_events)
[array([40.])]
>>> print(sol.t)
[0.00000000e+00 9.99900010e-05 1.09989001e-03 1.10988901e-02
 1.11088891e-01 1.11098890e+00 1.11099890e+01 4.00000000e+01]
```

---

## Piecewise polynomials and splines#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/interpolate/splines_and_polynomials.html

**Contents:**
- Piecewise polynomials and splines#
- Manipulating PPoly objects#
- B-splines: knots and coefficients#
  - B-spline basis elements#
  - Design matrices in the B-spline basis#
- Bernstein polynomials, BPoly#
- Conversion between bases#

1D interpolation routines discussed in the previous section, work by constructing certain piecewise polynomials: the interpolation range is split into intervals by the so-called breakpoints, and there is a certain polynomial on each interval. These polynomial pieces then match at the breakpoints with a predefined smoothness: the second derivatives for cubic splines, the first derivatives for monotone interpolants and so on.

A polynomial of degree \(k\) can be thought of as a linear combination of \(k+1\) monomial basis elements, \(1, x, x^2, \cdots, x^k\). In some applications, it is useful to consider alternative (if formally equivalent) bases. Two popular bases, implemented in scipy.interpolate are B-splines (BSpline) and Bernstein polynomials (BPoly). B-splines are often used for, for example, non-parametric regression problems, and Bernstein polynomials are used for constructing Bezier curves.

PPoly objects represent piecewise polynomials in the ‘usual’ power basis. This is the case for CubicSpline instances and monotone interpolants. In general, PPoly objects can represent polynomials of arbitrary orders, not only cubics. For the data array x, breakpoints are at the data points, and the array of coefficients, c , define polynomials of degree \(k\), such that c[i, j] is a coefficient for (x - x[j])**(k-i) on the segment between x[j] and x[j+1] .

BSpline objects represent B-spline functions — linear combinations of b-spline basis elements. These objects can be instantiated directly or constructed from data with the make_interp_spline factory function.

Finally, Bernstein polynomials are represented as instances of the BPoly class.

All these classes implement a (mostly) similar interface, PPoly being the most feature-complete. We next consider the main features of this interface and discuss some details of the alternative bases for piecewise polynomials.

PPoly objects have convenient methods for constructing derivatives and antiderivatives, computing integrals and root-finding. For example, we tabulate the sine function and find the roots of its derivative.

Now, differentiate the spline:

Here dspl is a PPoly instance which represents a polynomial approximation to the derivative of the original object, spl . Evaluating dspl at a fixed argument is equivalent to evaluating the original spline with the nu=1 argument:

Note that the second form above evaluates the derivative in place, while with the dspl object, we can find the zeros of the derivative of spl:

This agrees well with roots \(\pi/2 + \pi\,n\) of \(\cos(x) = \sin'(x)\). Note that by default it computed the roots extrapolated to the outside of the interpolation interval \(0 \leqslant x \leqslant 10\), and that the extrapolated results (the first and last values) are much less accurate. We can switch off the extrapolation and limit the root-finding to the interpolation interval:

In fact, the root method is a special case of a more general solve method which finds for a given constant \(y\) the solutions of the equation \(f(x) = y\) , where \(f(x)\) is the piecewise polynomial:

which agrees well with the expected values of \(\pm\arccos(1/2) + 2\pi\,n\).

Integrals of piecewise polynomials can be computed using the .integrate method which accepts the lower and the upper limits of integration. As an example, we compute an approximation to the complete elliptic integral \(K(m) = \int_0^{\pi/2} [1 - m\sin^2 x]^{-1/2} dx\):

To this end, we tabulate the integrand and interpolate it using the monotone PCHIP interpolant (we could as well used a CubicSpline):

which is indeed close to the value computed by scipy.special.ellipk.

All piecewise polynomials can be constructed with N-dimensional y values. If y.ndim > 1, it is understood as a stack of 1D y values, which are arranged along the interpolation axis (with the default value of 0). The latter is specified via the axis argument, and the invariant is that len(x) == y.shape[axis]. As an example, we extend the elliptic integral example above to compute the approximation for a range of m values, using the NumPy broadcasting:

Now the y array has the shape (11, 70), so that the values of y for fixed value of m are along the second axis of the y array.

A b-spline function — for instance, constructed from data via a make_interp_spline call — is defined by the so-called knots and coefficients.

As an illustration, let us again construct the interpolation of a sine function. The knots are available as the t attribute of a BSpline instance:

We see that the knot vector by default is constructed from the input array x: first, it is made \((k+1)\) -regular (it has k repeated knots appended and prepended); then, the second and second-to-last points of the input array are removed—this is the so-called not-a-knot boundary condition.

In general, an interpolating spline of degree k needs len(t) - len(x) - k - 1 boundary conditions. For cubic splines with (k+1)-regular knot arrays this means two boundary conditions—or removing two values from the x array. Various boundary conditions can be requested using the optional bc_type argument of make_interp_spline.

The b-spline coefficients are accessed via the c attribute of a BSpline object:

The convention is that for len(t) knots there are len(t) - k - 1 coefficients. Some routines (see the Smoothing splines section) zero-pad the c arrays so that len(c) == len(t). These additional coefficients are ignored for evaluation.

We stress that the coefficients are given in the b-spline basis, not the power basis of \(1, x, \cdots, x^k\).

The b-spline basis is used in a variety of applications which include interpolation, regression and curve representation. B-splines are piecewise polynomials, represented as linear combinations of b-spline basis elements — which themselves are certain linear combinations of usual monomials, \(x^m\) with \(m=0, 1, \dots, k\).

The properties of b-splines are well described in the literature (see, for example, references listed in the BSpline docstring). For our purposes, it is enough to know that a b-spline function is uniquely defined by an array of coefficients and an array of the so-called knots, which may or may not coincide with the data points, x.

Specifically, a b-spline basis element of degree k (e.g. k=3 for cubics) is defined by \(k+2\) knots and is zero outside of these knots. To illustrate, plot a collection of non-zero basis elements on a certain interval:

Here BSpline.basis_element is essentially a shorthand for constructing a spline with only a single non-zero coefficient. For instance, the j=2 element in the above example is equivalent to

If desired, a b-spline can be converted into a PPoly object using PPoly.from_spline method which accepts a BSpline instance and returns a PPoly instance. The reverse conversion is performed by the BSpline.from_power_basis method. However, conversions between bases is best avoided because it accumulates rounding errors.

One common application of b-splines is in non-parametric regression. The reason is that the localized nature of the b-spline basis elements makes linear algebra banded. This is because at most \(k+1\) basis elements are non-zero at a given evaluation point, thus a design matrix built on b-splines has at most \(k+1\) diagonals.

As an illustration, we consider a toy example. Suppose our data are one-dimensional and are confined to an interval \([0, 6]\). We construct a 4-regular knot vector which corresponds to 7 data points and cubic, k=3, splines:

Next, take ‘observations’ to be

and construct the design matrix in the sparse CSR format

Here each row of the design matrix corresponds to a value in the xnew array, and a row has no more than k+1 = 4 non-zero elements; row j contains basis elements evaluated at xnew[j]:

For \(t \in [0, 1]\), Bernstein basis polynomials of degree \(k\) are defined via

where \(C_k^a\) is the binomial coefficient, and \(a=0, 1, \dots, k\), so that there are \(k+1\) basis polynomials of degree \(k\).

A BPoly object represents a piecewise Bernstein polynomial in terms of breakpoints, x, and coefficients, c: c[a, j] gives the coefficient for \(b(t; k, a)\) for t on the interval between x[j] and x[j+1].

The user interface of BPoly objects is very similar to that of PPoly objects: both can be evaluated, differentiated and integrated.

One additional feature of BPoly objects is the alternative constructor, BPoly.from_derivatives, which constructs a BPoly object from data values and derivatives. Specifically, b = BPoly.from_derivatives(x, y) returns a callable that interpolates the provided values, b(x[i]) == y[i]), and has the provided derivatives, b(x[i], nu=j) == y[i][j].

This operation is similar to CubicHermiteSpline, but it is more flexible in that it can handle varying numbers of derivatives at different data points; i.e., the y argument can be a list of arrays of different lengths. See BPoly.from_derivatives for further discussion and examples.

In principle, all three bases for piecewise polynomials (the power basis, the Bernstein basis, and b-splines) are equivalent, and a polynomial in one basis can be converted into a different basis. One reason for converting between bases is that not all bases implement all operations. For instance, root-finding is only implemented for PPoly, and therefore to find roots of a BSpline object, you need to convert to PPoly first. See methods PPoly.from_bernstein_basis, PPoly.from_spline, BPoly.from_power_basis, and BSpline.from_power_basis for details about conversion.

In floating-point arithmetic, though, conversions always incur some precision loss. Whether this is significant is problem-dependent, so it is therefore recommended to exercise caution when converting between bases.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.interpolate import CubicSpline
>>> x = np.linspace(0, 10, 71)
>>> y = np.sin(x)
>>> spl = CubicSpline(x, y)
```

Example 2 (unknown):
```unknown
>>> dspl = spl.derivative()
```

Example 3 (unknown):
```unknown
>>> dspl(1.1), spl(1.1, nu=1)
(0.45361436, 0.45361436)
```

Example 4 (unknown):
```unknown
>>> dspl.roots() / np.pi
array([-0.45480801,  0.50000034,  1.50000099,  2.5000016 ,  3.46249993])
```

---

## NearestNDInterpolator#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html

**Contents:**
- NearestNDInterpolator#

Nearest-neighbor interpolator in N > 1 dimensions.

Data point coordinates.

Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.

Added in version 0.14.0.

Options passed to the underlying cKDTree.

Added in version 0.17.0.

__call__(*args, **query_options)

Evaluate interpolator at given points.

Interpolate unstructured D-D data.

Piecewise linear interpolator in N dimensions.

Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.

Interpolation on a regular grid or rectilinear grid.

Interpolator on a regular or rectilinear grid in arbitrary dimensions (interpn wraps this class).

Uses scipy.spatial.cKDTree

For data on a regular grid use interpn instead.

We can interpolate values on a 2D plane:

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.interpolate import NearestNDInterpolator
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> x = rng.random(10) - 0.5
>>> y = rng.random(10) - 0.5
>>> z = np.hypot(x, y)
>>> X = np.linspace(min(x), max(x))
>>> Y = np.linspace(min(y), max(y))
>>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
>>> interp = NearestNDInterpolator(list(zip(x, y)), z)
>>> Z = interp(X, Y)
>>> plt.pcolormesh(X, Y, Z, shading='auto')
>>> plt.plot(x, y, "ok", label="input point")
>>> plt.legend()
>>> plt.colorbar()
>>> plt.axis("equal")
>>> plt.show()
```

---

## tanm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.tanm.html

**Contents:**
- tanm#

Compute the matrix tangent.

This routine uses expm to compute the matrix exponentials.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Verify tanm(a) = sinm(a).dot(inv(cosm(a)))

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import tanm, sinm, cosm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> t = tanm(a)
>>> t
array([[ -2.00876993,  -8.41880636],
       [ -2.80626879, -10.42757629]])
```

Example 2 (json):
```json
>>> s = sinm(a)
>>> c = cosm(a)
>>> s.dot(np.linalg.inv(c))
array([[ -2.00876993,  -8.41880636],
       [ -2.80626879, -10.42757629]])
```

---

## schur#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.schur.html

**Contents:**
- schur#

Compute Schur decomposition of a matrix.

The Schur decomposition is:

where Z is unitary and T is either upper-triangular, or for real Schur decomposition (output=’real’), quasi-upper triangular. In the quasi-triangular form, 2x2 blocks describing complex-valued eigenvalue pairs may extrude from the diagonal.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

When the dtype of a is real, this specifies whether to compute the real or complex Schur decomposition. When the dtype of a is complex, this argument is ignored, and the complex Schur decomposition is computed.

Work array size. If None or -1, it is automatically computed.

Whether to overwrite data in a (may improve performance).

Specifies whether the upper eigenvalues should be sorted. A callable may be passed that, given an eigenvalue, returns a boolean denoting whether the eigenvalue should be sorted to the top-left (True).

If output='complex' OR the dtype of a is complex, the callable should have one argument: the eigenvalue expressed as a complex number.

If output='real' AND the dtype of a is real, the callable should have two arguments: the real and imaginary parts of the eigenvalue, respectively.

Alternatively, string parameters may be used:

Defaults to None (no sorting).

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Schur form of A. It is real-valued for the real Schur decomposition.

An unitary Schur transformation matrix for A. It is real-valued for the real Schur decomposition.

If and only if sorting was requested, a third return value will contain the number of eigenvalues satisfying the sort condition. Note that complex conjugate pairs for which the condition is true for either eigenvalue count as 2.

Error raised under three conditions:

The algorithm failed due to a failure of the QR algorithm to compute all eigenvalues.

If eigenvalue sorting was requested, the eigenvalues could not be reordered due to a failure to separate eigenvalues, usually because of poor conditioning.

If eigenvalue sorting was requested, roundoff errors caused the leading eigenvalues to no longer satisfy the sorting condition.

Convert real Schur form to complex Schur form

A custom eigenvalue-sorting condition that sorts by positive imaginary part is satisfied by only one eigenvalue.

When output='real' and the array a is real, the sort callable must accept the real and imaginary parts as separate arguments. Note that now the complex eigenvalues -0.32948354+0.80225456j and -0.32948354-0.80225456j will be treated as a complex conjugate pair, and according to the sdim documentation, complex conjugate pairs for which the condition is True for either eigenvalue increase sdim by two.

**Examples:**

Example 1 (unknown):
```unknown
A = Z T Z^H
```

Example 2 (unknown):
```unknown
'lhp'   Left-hand plane (real(eigenvalue) < 0.0)
'rhp'   Right-hand plane (real(eigenvalue) >= 0.0)
'iuc'   Inside the unit circle (abs(eigenvalue) <= 1.0)
'ouc'   Outside the unit circle (abs(eigenvalue) > 1.0)
```

Example 3 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import schur, eigvals
>>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
>>> T, Z = schur(A)
>>> T
array([[ 2.65896708,  1.42440458, -1.92933439],
       [ 0.        , -0.32948354, -0.49063704],
       [ 0.        ,  1.31178921, -0.32948354]])
>>> Z
array([[0.72711591, -0.60156188, 0.33079564],
       [0.52839428, 0.79801892, 0.28976765],
       [0.43829436, 0.03590414, -0.89811411]])
```

Example 4 (json):
```json
>>> T2, Z2 = schur(A, output='complex')
>>> T2
array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j], # may vary
       [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],
       [ 0.        ,  0.                    , -0.32948354-0.80225456j]])
>>> eigvals(T2)
array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])   # may vary
```

---

## Optimization (scipy.optimize)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/optimize.html

**Contents:**
- Optimization (scipy.optimize)#
- Local minimization of multivariate scalar functions (minimize)#
  - Unconstrained minimization#
    - Nelder-Mead Simplex algorithm (method='Nelder-Mead')#
    - Broyden-Fletcher-Goldfarb-Shanno algorithm (method='BFGS')#
    - Newton-Conjugate-Gradient algorithm (method='Newton-CG')#
    - Trust-Region Newton-Conjugate-Gradient Algorithm (method='trust-ncg')#
    - Trust-Region Truncated Generalized Lanczos / Conjugate Gradient Algorithm (method='trust-krylov')#
    - Trust-Region Nearly Exact Algorithm (method='trust-exact')#
  - Constrained minimization#

Optimization (scipy.optimize)

Local minimization of multivariate scalar functions (minimize)

Unconstrained minimization

Nelder-Mead Simplex algorithm (method='Nelder-Mead')

Broyden-Fletcher-Goldfarb-Shanno algorithm (method='BFGS')

Newton-Conjugate-Gradient algorithm (method='Newton-CG')

Trust-Region Newton-Conjugate-Gradient Algorithm (method='trust-ncg')

Trust-Region Truncated Generalized Lanczos / Conjugate Gradient Algorithm (method='trust-krylov')

Trust-Region Nearly Exact Algorithm (method='trust-exact')

Constrained minimization

Trust-Region Constrained Algorithm (method='trust-constr')

Sequential Least SQuares Programming (SLSQP) Algorithm (method='SLSQP')

Local minimization solver comparison

Comparison of Global Optimizers

Least-squares minimization (least_squares)

Example of solving a fitting problem

Univariate function minimizers (minimize_scalar)

Unconstrained minimization (method='brent')

Bounded minimization (method='bounded')

Root finding for large problems

Still too slow? Preconditioning.

Linear programming (linprog)

Linear programming example

Linear sum assignment problem example

Mixed integer linear programming

Knapsack problem example

Parallel execution support

The scipy.optimize package provides several commonly used optimization algorithms. A detailed listing is available: scipy.optimize (can also be found by help(scipy.optimize)).

The minimize function provides a common interface to unconstrained and constrained minimization algorithms for multivariate scalar functions in scipy.optimize. To demonstrate the minimization function, consider the problem of minimizing the Rosenbrock function of \(N\) variables:

The minimum value of this function is 0 which is achieved when \(x_{i}=1.\)

Note that the Rosenbrock function and its derivatives are included in scipy.optimize. The implementations shown in the following sections provide examples of how to define an objective function as well as its jacobian and hessian functions. Objective functions in scipy.optimize expect a numpy array as their first parameter which is to be optimized and must return a float value. The exact calling signature must be f(x, *args) where x represents a numpy array and args a tuple of additional arguments supplied to the objective function.

In the example below, the minimize routine is used with the Nelder-Mead simplex algorithm (selected through the method parameter):

The simplex algorithm is probably the simplest way to minimize a fairly well-behaved function. It requires only function evaluations and is a good choice for simple minimization problems. However, because it does not use any gradient evaluations, it may take longer to find the minimum.

Another optimization algorithm that needs only function calls to find the minimum is Powell’s method available by setting method='powell' in minimize.

To demonstrate how to supply additional arguments to an objective function, let us minimize the Rosenbrock function with an additional scaling factor a and an offset b:

Again using the minimize routine this can be solved by the following code block for the example parameters a=0.5 and b=1.

As an alternative to using the args parameter of minimize, simply wrap the objective function in a new function that accepts only x. This approach is also useful when it is necessary to pass additional parameters to the objective function as keyword arguments.

Another alternative is to use functools.partial.

In order to converge more quickly to the solution, this routine uses the gradient of the objective function. If the gradient is not given by the user, then it is estimated using first-differences. The Broyden-Fletcher-Goldfarb-Shanno (BFGS) method typically requires fewer function calls than the simplex algorithm even when the gradient must be estimated.

To demonstrate this algorithm, the Rosenbrock function is again used. The gradient of the Rosenbrock function is the vector:

This expression is valid for the interior derivatives. Special cases are

A Python function which computes this gradient is constructed by the code-segment:

This gradient information is specified in the minimize function through the jac parameter as illustrated below.

Avoiding Redundant Calculation

It is common for the objective function and its gradient to share parts of the calculation. For instance, consider the following problem.

Here, expensive is called 12 times: six times in the objective function and six times from the gradient. One way of reducing redundant calculations is to create a single function that returns both the objective function and the gradient.

When we call minimize, we specify jac==True to indicate that the provided function returns both the objective function and its gradient. While convenient, not all scipy.optimize functions support this feature, and moreover, it is only for sharing calculations between the function and its gradient, whereas in some problems we will want to share calculations with the Hessian (second derivative of the objective function) and constraints. A more general approach is to memoize the expensive parts of the calculation. In simple situations, this can be accomplished with the functools.lru_cache wrapper.

Newton-Conjugate Gradient algorithm is a modified Newton’s method and uses a conjugate gradient algorithm to (approximately) invert the local Hessian [NW]. Newton’s method is based on fitting the function locally to a quadratic form:

where \(\mathbf{H}\left(\mathbf{x}_{0}\right)\) is a matrix of second-derivatives (the Hessian). If the Hessian is positive definite then the local minimum of this function can be found by setting the gradient of the quadratic form to zero, resulting in

The inverse of the Hessian is evaluated using the conjugate-gradient method. An example of employing this method to minimizing the Rosenbrock function is given below. To take full advantage of the Newton-CG method, a function which computes the Hessian must be provided. The Hessian matrix itself does not need to be constructed, only a vector which is the product of the Hessian with an arbitrary vector needs to be available to the minimization routine. As a result, the user can provide either a function to compute the Hessian matrix, or a function to compute the product of the Hessian with an arbitrary vector.

The Hessian of the Rosenbrock function is

if \(i,j\in\left[1,N-2\right]\) with \(i,j\in\left[0,N-1\right]\) defining the \(N\times N\) matrix. Other non-zero entries of the matrix are

For example, the Hessian when \(N=5\) is

The code which computes this Hessian along with the code to minimize the function using Newton-CG method is shown in the following example:

Hessian product example

For larger minimization problems, storing the entire Hessian matrix can consume considerable time and memory. The Newton-CG algorithm only needs the product of the Hessian times an arbitrary vector. As a result, the user can supply code to compute this product rather than the full Hessian by giving a hess function which take the minimization vector as the first argument and the arbitrary vector as the second argument (along with extra arguments passed to the function to be minimized). If possible, using Newton-CG with the Hessian product option is probably the fastest way to minimize the function.

In this case, the product of the Rosenbrock Hessian with an arbitrary vector is not difficult to compute. If \(\mathbf{p}\) is the arbitrary vector, then \(\mathbf{H}\left(\mathbf{x}\right)\mathbf{p}\) has elements:

Code which makes use of this Hessian product to minimize the Rosenbrock function using minimize follows:

According to [NW] p. 170 the Newton-CG algorithm can be inefficient when the Hessian is ill-conditioned because of the poor quality search directions provided by the method in those situations. The method trust-ncg, according to the authors, deals more effectively with this problematic situation and will be described next.

The Newton-CG method is a line search method: it finds a direction of search minimizing a quadratic approximation of the function and then uses a line search algorithm to find the (nearly) optimal step size in that direction. An alternative approach is to, first, fix the step size limit \(\Delta\) and then find the optimal step \(\mathbf{p}\) inside the given trust-radius by solving the following quadratic subproblem:

The solution is then updated \(\mathbf{x}_{k+1} = \mathbf{x}_{k} + \mathbf{p}\) and the trust-radius \(\Delta\) is adjusted according to the degree of agreement of the quadratic model with the real function. This family of methods is known as trust-region methods. The trust-ncg algorithm is a trust-region method that uses a conjugate gradient algorithm to solve the trust-region subproblem [NW].

Hessian product example

Similar to the trust-ncg method, the trust-krylov method is a method suitable for large-scale problems as it uses the hessian only as linear operator by means of matrix-vector products. It solves the quadratic subproblem more accurately than the trust-ncg method.

This method wraps the [TRLIB] implementation of the [GLTR] method solving exactly a trust-region subproblem restricted to a truncated Krylov subspace. For indefinite problems it is usually better to use this method as it reduces the number of nonlinear iterations at the expense of few more matrix-vector products per subproblem solve in comparison to the trust-ncg method.

Hessian product example

F. Lenders, C. Kirches, A. Potschka: “trlib: A vector-free implementation of the GLTR method for iterative solution of the trust region problem”, arXiv:1611.04718

N. Gould, S. Lucidi, M. Roma, P. Toint: “Solving the Trust-Region Subproblem using the Lanczos Method”, SIAM J. Optim., 9(2), 504–525, (1999). DOI:10.1137/S1052623497322735

All methods Newton-CG, trust-ncg and trust-krylov are suitable for dealing with large-scale problems (problems with thousands of variables). That is because the conjugate gradient algorithm approximately solve the trust-region subproblem (or invert the Hessian) by iterations without the explicit Hessian factorization. Since only the product of the Hessian with an arbitrary vector is needed, the algorithm is specially suited for dealing with sparse Hessians, allowing low storage requirements and significant time savings for those sparse problems.

For medium-size problems, for which the storage and factorization cost of the Hessian are not critical, it is possible to obtain a solution within fewer iteration by solving the trust-region subproblems almost exactly. To achieve that, a certain nonlinear equations is solved iteratively for each quadratic subproblem [CGT]. This solution requires usually 3 or 4 Cholesky factorizations of the Hessian matrix. As the result, the method converges in fewer number of iterations and takes fewer evaluations of the objective function than the other implemented trust-region methods. The Hessian product option is not supported by this algorithm. An example using the Rosenbrock function follows:

J. Nocedal, S.J. Wright “Numerical optimization.” 2nd edition. Springer Science (2006).

Conn, A. R., Gould, N. I., & Toint, P. L. “Trust region methods”. Siam. (2000). pp. 169-200.

The minimize function provides several algorithms for constrained minimization, namely 'trust-constr' , 'SLSQP', 'COBYLA', and 'COBYQA'. They require the constraints to be defined using slightly different structures. The methods 'trust-constr', 'COBYQA', and 'COBYLA' require the constraints to be defined as a sequence of objects LinearConstraint and NonlinearConstraint. Method 'SLSQP', on the other hand, requires constraints to be defined as a sequence of dictionaries, with keys type, fun and jac.

As an example let us consider the constrained minimization of the Rosenbrock function:

This optimization problem has the unique solution \([x_0, x_1] = [0.4149,~ 0.1701]\), for which only the first and fourth constraints are active.

The trust-region constrained method deals with constrained minimization problems of the form:

When \(c^l_j = c^u_j\) the method reads the \(j\)-th constraint as an equality constraint and deals with it accordingly. Besides that, one-sided constraint can be specified by setting the upper or lower bound to np.inf with the appropriate sign.

The implementation is based on [EQSQP] for equality-constraint problems and on [TRIP] for problems with inequality constraints. Both are trust-region type algorithms suitable for large-scale problems.

Defining Bounds Constraints

The bound constraints \(0 \leq x_0 \leq 1\) and \(-0.5 \leq x_1 \leq 2.0\) are defined using a Bounds object.

Defining Linear Constraints

The constraints \(x_0 + 2 x_1 \leq 1\) and \(2 x_0 + x_1 = 1\) can be written in the linear constraint standard format:

and defined using a LinearConstraint object.

Defining Nonlinear Constraints The nonlinear constraint:

with Jacobian matrix:

and linear combination of the Hessians:

is defined using a NonlinearConstraint object.

Alternatively, it is also possible to define the Hessian \(H(x, v)\) as a sparse matrix,

or as a LinearOperator object.

When the evaluation of the Hessian \(H(x, v)\) is difficult to implement or computationally infeasible, one may use HessianUpdateStrategy. Currently available strategies are BFGS and SR1.

Alternatively, the Hessian may be approximated using finite differences.

The Jacobian of the constraints can be approximated by finite differences as well. In this case, however, the Hessian cannot be computed with finite differences and needs to be provided by the user or defined using HessianUpdateStrategy.

Solving the Optimization Problem The optimization problem is solved using:

When needed, the objective function Hessian can be defined using a LinearOperator object,

or a Hessian-vector product through the parameter hessp.

Alternatively, the first and second derivatives of the objective function can be approximated. For instance, the Hessian can be approximated with SR1 quasi-Newton approximation and the gradient with finite differences.

Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal. 1999. An interior point algorithm for large-scale nonlinear programming. SIAM Journal on Optimization 9.4: 877-900.

Lalee, Marucha, Jorge Nocedal, and Todd Plantega. 1998. On the implementation of an algorithm for large-scale equality constrained optimization. SIAM Journal on Optimization 8.3: 682-706.

The SLSQP method deals with constrained minimization problems of the form:

Where \(\mathcal{E}\) or \(\mathcal{I}\) are sets of indices containing equality and inequality constraints.

Both linear and nonlinear constraints are defined as dictionaries with keys type, fun and jac.

And the optimization problem is solved with:

Most of the options available for the method 'trust-constr' are not available for 'SLSQP'.

Find a solver that meets your requirements using the table below. If there are multiple candidates, try several and see which ones best meet your needs (e.g. execution time, objective function value).

Nonlinear Constraints

Global optimization aims to find the global minimum of a function within given bounds, in the presence of potentially many local minima. Typically, global minimizers efficiently search the parameter space, while using a local minimizer (e.g., minimize) under the hood. SciPy contains a number of good global optimizers. Here, we’ll use those on the same objective function, namely the (aptly named) eggholder function:

This function looks like an egg carton:

We now use the global optimizers to obtain the minimum and the function value at the minimum. We’ll store the results in a dictionary so we can compare different optimization results later.

All optimizers return an OptimizeResult, which in addition to the solution contains information on the number of function evaluations, whether the optimization was successful, and more. For brevity, we won’t show the full output of the other optimizers:

shgo has a second method, which returns all local minima rather than only what it thinks is the global minimum:

We’ll now plot all found minima on a heatmap of the function:

Find a solver that meets your requirements using the table below. If there are multiple candidates, try several and see which ones best meet your needs (e.g. execution time, objective function value).

Nonlinear Constraints

differential_evolution

(✓) = Depending on the chosen local minimizer

SciPy is capable of solving robustified bound-constrained nonlinear least-squares problems:

Here \(f_i(\mathbf{x})\) are smooth functions from \(\mathbb{R}^n\) to \(\mathbb{R}\), we refer to them as residuals. The purpose of a scalar-valued function \(\rho(\cdot)\) is to reduce the influence of outlier residuals and contribute to robustness of the solution, we refer to it as a loss function. A linear loss function gives a standard least-squares problem. Additionally, constraints in a form of lower and upper bounds on some of \(x_j\) are allowed.

All methods specific to least-squares minimization utilize a \(m \times n\) matrix of partial derivatives called Jacobian and defined as \(J_{ij} = \partial f_i / \partial x_j\). It is highly recommended to compute this matrix analytically and pass it to least_squares, otherwise, it will be estimated by finite differences, which takes a lot of additional time and can be very inaccurate in hard cases.

Function least_squares can be used for fitting a function \(\varphi(t; \mathbf{x})\) to empirical data \(\{(t_i, y_i), i = 0, \ldots, m-1\}\). To do this, one should simply precompute residuals as \(f_i(\mathbf{x}) = w_i (\varphi(t_i; \mathbf{x}) - y_i)\), where \(w_i\) are weights assigned to each observation.

Here we consider an enzymatic reaction [1]. There are 11 residuals defined as

where \(y_i\) are measurement values and \(u_i\) are values of the independent variable. The unknown vector of parameters is \(\mathbf{x} = (x_0, x_1, x_2, x_3)^T\). As was said previously, it is recommended to compute Jacobian matrix in a closed form:

We are going to use the “hard” starting point defined in [2]. To find a physically meaningful solution, avoid potential division by zero and assure convergence to the global minimum we impose constraints \(0 \leq x_j \leq 100, j = 0, 1, 2, 3\).

The code below implements least-squares estimation of \(\mathbf{x}\) and finally plots the original data and the fitted model function:

J. Kowalik and J. F. Morrison, “Analysis of kinetic data for allosteric enzyme reactions as a nonlinear regression problem”, Math. Biosci., vol. 2, pp. 57-66, 1968.

Averick et al., “The MINPACK-2 Test Problem Collection”.

Three interactive examples below illustrate usage of least_squares in greater detail.

Large-scale bundle adjustment in scipy demonstrates large-scale capabilities of least_squares and how to efficiently compute finite difference approximation of sparse Jacobian.

Robust nonlinear regression in scipy shows how to handle outliers with a robust loss function in a nonlinear regression.

Solving a discrete boundary-value problem in scipy examines how to solve a large system of equations and use bounds to achieve desired properties of the solution.

For the details about mathematical algorithms behind the implementation refer to documentation of least_squares.

Often only the minimum of an univariate function (i.e., a function that takes a scalar as input) is needed. In these circumstances, other optimization techniques have been developed that can work faster. These are accessible from the minimize_scalar function, which proposes several algorithms.

There are, actually, two methods that can be used to minimize an univariate function: brent and golden, but golden is included only for academic purposes and should rarely be used. These can be respectively selected through the method parameter in minimize_scalar. The brent method uses Brent’s algorithm for locating a minimum. Optimally, a bracket (the bracket parameter) should be given which contains the minimum desired. A bracket is a triple \(\left( a, b, c \right)\) such that \(f \left( a \right) > f \left( b \right) < f \left( c \right)\) and \(a < b < c\) . If this is not given, then alternatively two starting points can be chosen and a bracket will be found from these points using a simple marching algorithm. If these two starting points are not provided, 0 and 1 will be used (this may not be the right choice for your function and result in an unexpected minimum being returned).

Very often, there are constraints that can be placed on the solution space before minimization occurs. The bounded method in minimize_scalar is an example of a constrained minimization procedure that provides a rudimentary interval constraint for scalar functions. The interval constraint allows the minimization to occur only between two fixed endpoints, specified using the mandatory bounds parameter.

For example, to find the minimum of \(J_{1}\left( x \right)\) near \(x=5\) , minimize_scalar can be called using the interval \(\left[ 4, 7 \right]\) as a constraint. The result is \(x_{\textrm{min}}=5.3314\) :

Sometimes, it may be useful to use a custom method as a (multivariate or univariate) minimizer, for example, when using some library wrappers of minimize (e.g., basinhopping).

We can achieve that by, instead of passing a method name, passing a callable (either a function or an object implementing a __call__ method) as the method parameter.

Let us consider an (admittedly rather virtual) need to use a trivial custom multivariate minimization method that will just search the neighborhood in each dimension independently with a fixed step size:

This will work just as well in case of univariate optimization:

If one has a single-variable equation, there are multiple different root finding algorithms that can be tried. Most of these algorithms require the endpoints of an interval in which a root is expected (because the function changes signs). In general, brentq is the best choice, but the other methods may be useful in certain circumstances or for academic purposes. When a bracket is not available, but one or more derivatives are available, then newton (or halley, secant) may be applicable. This is especially the case if the function is defined on a subset of the complex plane, and the bracketing methods cannot be used.

A problem closely related to finding the zeros of a function is the problem of finding a fixed point of a function. A fixed point of a function is the point at which evaluation of the function returns the point: \(g\left(x\right)=x.\) Clearly, the fixed point of \(g\) is the root of \(f\left(x\right)=g\left(x\right)-x.\) Equivalently, the root of \(f\) is the fixed point of \(g\left(x\right)=f\left(x\right)+x.\) The routine fixed_point provides a simple iterative method using Aitkens sequence acceleration to estimate the fixed point of \(g\) given a starting point.

Finding a root of a set of non-linear equations can be achieved using the root function. Several methods are available, amongst which hybr (the default) and lm, which, respectively, use the hybrid method of Powell and the Levenberg-Marquardt method from MINPACK.

The following example considers the single-variable transcendental equation

a root of which can be found as follows:

Consider now a set of non-linear equations

We define the objective function so that it also returns the Jacobian and indicate this by setting the jac parameter to True. Also, the Levenberg-Marquardt solver is used here.

Methods hybr and lm in root cannot deal with a very large number of variables (N), as they need to calculate and invert a dense N x N Jacobian matrix on every Newton step. This becomes rather inefficient when N grows.

Consider, for instance, the following problem: we need to solve the following integrodifferential equation on the square \([0,1]\times[0,1]\):

with the boundary condition \(P(x,1) = 1\) on the upper edge and \(P=0\) elsewhere on the boundary of the square. This can be done by approximating the continuous function P by its values on a grid, \(P_{n,m}\approx{}P(n h, m h)\), with a small grid spacing h. The derivatives and integrals can then be approximated; for instance \(\partial_x^2 P(x,y)\approx{}(P(x+h,y) - 2 P(x,y) + P(x-h,y))/h^2\). The problem is then equivalent to finding the root of some function residual(P), where P is a vector of length \(N_x N_y\).

Now, because \(N_x N_y\) can be large, methods hybr or lm in root will take a long time to solve this problem. The solution can, however, be found using one of the large-scale solvers, for example krylov, broyden2, or anderson. These use what is known as the inexact Newton method, which instead of computing the Jacobian matrix exactly, forms an approximation for it.

The problem we have can now be solved as follows:

When looking for the zero of the functions \(f_i({\bf x}) = 0\), i = 1, 2, …, N, the krylov solver spends most of the time inverting the Jacobian matrix,

If you have an approximation for the inverse matrix \(M\approx{}J^{-1}\), you can use it for preconditioning the linear-inversion problem. The idea is that instead of solving \(J{\bf s}={\bf y}\) one solves \(MJ{\bf s}=M{\bf y}\): since matrix \(MJ\) is “closer” to the identity matrix than \(J\) is, the equation should be easier for the Krylov method to deal with.

The matrix M can be passed to root with method krylov as an option options['jac_options']['inner_M']. It can be a (sparse) matrix or a scipy.sparse.linalg.LinearOperator instance.

For the problem in the previous section, we note that the function to solve consists of two parts: the first one is the application of the Laplace operator, \([\partial_x^2 + \partial_y^2] P\), and the second is the integral. We can actually easily compute the Jacobian corresponding to the Laplace operator part: we know that in 1-D

so that the whole 2-D operator is represented by

The matrix \(J_2\) of the Jacobian corresponding to the integral is more difficult to calculate, and since all of it entries are nonzero, it will be difficult to invert. \(J_1\) on the other hand is a relatively simple matrix, and can be inverted by scipy.sparse.linalg.splu (or the inverse can be approximated by scipy.sparse.linalg.spilu). So we are content to take \(M\approx{}J_1^{-1}\) and hope for the best.

In the example below, we use the preconditioner \(M=J_1^{-1}\).

Resulting run, first without preconditioning:

and then with preconditioning:

Using a preconditioner reduced the number of evaluations of the residual function by a factor of 4. For problems where the residual is expensive to compute, good preconditioning can be crucial — it can even decide whether the problem is solvable in practice or not.

Preconditioning is an art, science, and industry. Here, we were lucky in making a simple choice that worked reasonably well, but there is a lot more depth to this topic than is shown here.

The function linprog can minimize a linear objective function subject to linear equality and inequality constraints. This kind of problem is well known as linear programming. Linear programming solves problems of the following form:

where \(x\) is a vector of decision variables; \(c\), \(b_{ub}\), \(b_{eq}\), \(l\), and \(u\) are vectors; and \(A_{ub}\) and \(A_{eq}\) are matrices.

In this tutorial, we will try to solve a typical linear programming problem using linprog.

Consider the following simple linear programming problem:

We need some mathematical manipulations to convert the target problem to the form accepted by linprog.

First of all, let’s consider the objective function. We want to maximize the objective function, but linprog can only accept a minimization problem. This is easily remedied by converting the maximize \(29x_1 + 45x_2\) to minimizing \(-29x_1 -45x_2\). Also, \(x_3, x_4\) are not shown in the objective function. That means the weights corresponding with \(x_3, x_4\) are zero. So, the objective function can be converted to:

If we define the vector of decision variables \(x = [x_1, x_2, x_3, x_4]^T\), the objective weights vector \(c\) of linprog in this problem should be

Next, let’s consider the two inequality constraints. The first one is a “less than” inequality, so it is already in the form accepted by linprog. The second one is a “greater than” inequality, so we need to multiply both sides by \(-1\) to convert it to a “less than” inequality. Explicitly showing zero coefficients, we have:

These equations can be converted to matrix form:

Next, let’s consider the two equality constraints. Showing zero weights explicitly, these are:

These equations can be converted to matrix form:

Lastly, let’s consider the separate inequality constraints on individual decision variables, which are known as “box constraints” or “simple bounds”. These constraints can be applied using the bounds argument of linprog. As noted in the linprog documentation, the default value of bounds is (0, None), meaning that the lower bound on each decision variable is 0, and the upper bound on each decision variable is infinity: all the decision variables are non-negative. Our bounds are different, so we will need to specify the lower and upper bound on each decision variable as a tuple and group these tuples into a list.

Finally, we can solve the transformed problem using linprog.

The result states that our problem is infeasible, meaning that there is no solution vector that satisfies all the constraints. That doesn’t necessarily mean we did anything wrong; some problems truly are infeasible. Suppose, however, that we were to decide that our bound constraint on \(x_1\) was too tight and that it could be loosened to \(0 \leq x_1 \leq 6\). After adjusting our code x1_bounds = (0, 6) to reflect the change and executing it again:

The result shows the optimization was successful. We can check the objective value (result.fun) is same as \(c^Tx\):

We can also check that all constraints are satisfied within reasonable tolerances:

Consider the problem of selecting students for a swimming medley relay team. We have a table showing times for each swimming style of five students:

We need to choose a student for each of the four swimming styles such that the total relay time is minimized. This is a typical linear sum assignment problem. We can use linear_sum_assignment to solve it.

The linear sum assignment problem is one of the most famous combinatorial optimization problems. Given a “cost matrix” \(C\), the problem is to choose

exactly one element from each row

without choosing more than one element from any column

such that the sum of the chosen elements is minimized

In other words, we need to assign each row to one column such that the sum of the corresponding entries is minimized.

Formally, let \(X\) be a boolean matrix where \(X[i,j] = 1\) iff row \(i\) is assigned to column \(j\). Then the optimal assignment has cost

The first step is to define the cost matrix. In this example, we want to assign each swimming style to a student. linear_sum_assignment is able to assign each row of a cost matrix to a column. Therefore, to form the cost matrix, the table above needs to be transposed so that the rows correspond with swimming styles and the columns correspond with students:

We can solve the assignment problem with linear_sum_assignment:

The row_ind and col_ind are optimal assigned matrix indexes of the cost matrix:

The optimal assignment is:

The optimal total medley time is:

Note that this result is not the same as the sum of the minimum times for each swimming style:

because student “C” is the best swimmer in both “breaststroke” and “butterfly” style. We cannot assign student “C” to both styles, so we assigned student C to the “breaststroke” style and D to the “butterfly” style to minimize the total time.

Some further reading and related software, such as Newton-Krylov [KK], PETSc [PP], and PyAMG [AMG]:

D.A. Knoll and D.E. Keyes, “Jacobian-free Newton-Krylov methods”, J. Comp. Phys. 193, 357 (2004). DOI:10.1016/j.jcp.2003.08.010

PETSc https://www.mcs.anl.gov/petsc/ and its Python bindings https://bitbucket.org/petsc/petsc4py/

PyAMG (algebraic multigrid preconditioners/solvers) pyamg/pyamg#issues

The knapsack problem is a well known combinatorial optimization problem. Given a set of items, each with a size and a value, the problem is to choose the items that maximize the total value under the condition that the total size is below a certain threshold.

\(x_i\) be a boolean variable that indicates whether item \(i\) is included in the knapsack,

\(n\) be the total number of items,

\(v_i\) be the value of item \(i\),

\(s_i\) be the size of item \(i\), and

\(C\) be the capacity of the knapsack.

Although the objective function and inequality constraints are linear in the decision variables \(x_i\), this differs from a typical linear programming problem in that the decision variables can only assume integer values. Specifically, our decision variables can only be \(0\) or \(1\), so this is known as a binary integer linear program (BILP). Such a problem falls within the larger class of mixed integer linear programs (MILPs), which we we can solve with milp.

In our example, there are 8 items to choose from, and the size and value of each is specified as follows.

We need to constrain our eight decision variables to be binary. We do so by adding a Bounds: constraint to ensure that they lie between \(0\) and \(1\), and we apply “integrality” constraints to ensure that they are either \(0\) or \(1\).

The knapsack capacity constraint is specified using LinearConstraint.

If we are following the usual rules of linear algebra, the input A should be a two-dimensional matrix, and the lower and upper bounds lb and ub should be one-dimensional vectors, but LinearConstraint is forgiving as long as the inputs can be broadcast to consistent shapes.

Using the variables defined above, we can solve the knapsack problem using milp. Note that milp minimizes the objective function, but we want to maximize the total value, so we set c to be negative of the values.

Let’s check the result:

This means that we should select the items 1, 2, 4, 5, 6 to optimize the total value under the size constraint. Note that this is different from we would have obtained had we solved the linear programming relaxation (without integrality constraints) and attempted to round the decision variables.

If we were to round this solution up to array([1., 1., 1., 1., 1., 1., 0., 0.]), our knapsack would be over the capacity constraint, whereas if we were to round down to array([1., 1., 1., 1., 0., 1., 0., 0.]), we would have a sub-optimal solution.

For more MILP tutorials, see the Jupyter notebooks on SciPy Cookbooks:

Compressed Sensing l1 program

Compressed Sensing l0 program

Some SciPy optimization methods, such as differential_evolution, offer parallelization through the use of a workers keyword.

For differential_evolution there are two loops (iteration) levels in the algorithm. The outer loop represents successive generations of a population. This loop can’t be parallelized. For a given generation candidate solutions are generated that have to be compared against existing population members. The fitness of the candidate solution can be done in a loop, but it’s also possible to parallelize the calculation.

Parallelization is also possible in other optimization algorithms. For example in various minimize methods numerical differentiation is used to estimate derivatives. For a simple gradient calculation using two-point forward differences a total of N + 1 objective function calculations have to be done, where N is the number of parameters. These are just small perturbations around a given location (the +1). Those N + 1 calculations are also parallelizable. The calculation of numerical derivatives are used by the minimization algorithm to generate new steps.

Each optimization algorithm is quite different in how they work, but they often have locations where multiple objective function calculations are required before the algorithm does something else. Those locations are what can be parallelized. There are therefore common characteristics in how workers is used. These commonalities are described below.

If an int is supplied then a multiprocessing.Pool is created, with the object’s map method being used to evaluate solutions in parallel. With this approach it is mandatory that the objective function is pickleable. Lambda functions do not meet that requirement.

It is also possible to use a map-like callable as a worker. Here the map-like function is provided with a series of vectors that the optimization algorithm provides. The map-like function needs to evaluate each vector against the objective function. In the following example we use multiprocessing.Pool as the map-like. As before, the objective function still needs to be pickleable. This example is semantically identical to the previous example.

It can be an advantage to use this pattern because the Pool can be re-used for further calculations - there is a significant amount of overhead in creating those objects. Alternatives to multiprocessing.Pool include the mpi4py package, which enables parallel processing on clusters.

In Scipy 1.16.0 the workers keyword was introduced to selected minimize methods. Here parallelization is typically applied during numerical differentiation. Either of the two approaches outlined above can be used, although it’s strongly advised to supply the map-like callable due to the overhead of creating new processes. Performance gains will only be made if the objective function is expensive to calculate. Let’s compare how much parallelization can help compared to the serial version. To simulate a slow function we use the time package.

Examine the serial minimization first:

Now the parallel version:

If the objective function can be vectorized, then a map-like can be used to take advantage of vectorization during function evaluation. Vectorization means that the objective function can carry out the required calculations in a single (rather than multiple) call, which is typically very efficient:

There are several important points to note about this example:

The iterable represents the series of parameter vectors that the algorithm wishes to be evaluated.

The iterable is first converted to an iterator, before being made into an array via a list comprehension. This allows the iterable to be a generator, list, array, etc.

Within the map-like the calculation is done using slow_func instead of using fun. The map-like is actually supplied with a wrapped version of the objective function. The wrapping is used to detect various types of common user errors, including checking whether the objective function returns a scalar. If fun is used then a RuntimeError will result, because fun(arr_t) will be a 1-D array and not a scalar. We therefore use slow_func directly.

arr.T is sent to the objective function. This is because arr.shape == (S, N), where S is the number of parameter vectors to evaluate and N is the number of variables. For slow_func vectorization occurs on (N, S) shaped arrays.

This approach is not needed for differential_evolution as that minimizer already has a keyword for vectorization.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.optimize import minimize
```

Example 2 (python):
```python
>>> def rosen(x):
...     """The Rosenbrock function"""
...     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
```

Example 3 (json):
```json
>>> x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
>>> res = minimize(rosen, x0, method='nelder-mead',
...                options={'xatol': 1e-8, 'disp': True})
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 339
         Function evaluations: 571
```

Example 4 (json):
```json
>>> print(res.x)
[1. 1. 1. 1. 1.]
```

---

## interp2d transition guide#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/interpolate/interp_transition_guide.html

**Contents:**
- interp2d transition guide#
- 1. How to transition away from using interp2d#
  - 1.1 interp2d on a regular grid#
    - Replacement: Use RectBivariateSpline, the result is identical#
    - Interpolation order: linear, cubic etc#
  - 1.2. interp2d with full coordinates of points (scattered interpolation)#
    - Replacement: Use scipy.interpolate.bisplrep / scipy.interpolate.bisplev directly#
    - Interpolation order: linear, cubic etc#
- 2. Alternative to interp2d: regular grid#
- 3. Scattered 2D linear interpolation: prefer LinearNDInterpolator to SmoothBivariateSpline or bisplrep#

This page contains three sets of demonstrations:

lower-level FITPACK replacements for scipy.interpolate.interp2d for legacy bug-for-bug compatible scipy.interpolate.interp2d replacements;

recommended replacements for scipy.interpolate.interp2d for use in new code;

a demonstration of failure modes of 2D FITPACK-based linear interpolation and recommended replacements.

interp2d silently switches between interpolation on a 2D regular grid and interpolating 2D scattered data. The switch is based on the lengths of the (raveled) x, y, and z arrays. In short, for regular grid use scipy.interpolate.RectBivariateSpline; for scattered interpolation, use the bisprep/bisplev combo. Below we give examples of the literal point-for-point transition, which should preserve the interp2d results exactly.

We start from the (slightly modified) docstring example.

This is the “regular grid” code path, because

Also, note that x.size != y.size:

Now, let’s build a convenience function to construct the interpolator and plot it.

Note the transposes: first, in the constructor, second, you need to transpose the result of the evaluation. This is to undo the transposes interp2d does.

interp2d defaults to kind="linear", which is linear in both directions, x- and y-. RectBivariateSpline, on the other hand, defaults to cubic interpolation, kx=3, ky=3.

Here is the exact equivalence:

Here, we flatten the meshgrid from the previous exercise to illustrate the functionality.

Note that this the “not regular grid” code path, meant for scattered data, with len(x) == len(y) == len(z).

interp2d defaults to kind="linear", which is linear in both directions, x- and y-. bisplrep, on the other hand, defaults to cubic interpolation, kx=3, ky=3.

Here is the exact equivalence:

For new code, the recommended alternative is RegularGridInterpolator. It is an independent implementation, not based on FITPACK. Supports nearest, linear interpolation and odd-order tensor product splines.

The spline knots are guaranteed to coincide with the data points.

the tuple argument, is (x, y)

z array needs a transpose

the keyword name is method, not kind

bounds_error argument is True by default.

Evaluation: create a 2D meshgrid. Use indexing=’ij’ and sparse=True to save some memory:

Evaluate, note the tuple argument:

For 2D scattered linear interpolation, both SmoothBivariateSpline and biplrep may either emit warnings, or fail to interpolate the data, or produce splines which with knots away from the data points. Instead, prefer LinearNDInterpolator, which is based on triangulating the data via QHull.

Now, use the linear interpolation over Qhull-based triangulation of data:

The result is easy to understand and interpret:

Note that bisplrep does something different! It may place spline knots outside of the data.

For illustration, consider the same data from the previous example:

Also, SmoothBivariateSpline fails to interpolate the data. Again, use the same data from the previous example.

Note that both SmoothBivariateSpline and bisplrep results have artifacts, unlike the LinearNDInterpolator’s. Issues illustrated here were reported for linear interpolation, however the FITPACK knot-selection mechanism does not guarantee to avoid either of these issues for higher-order (e.g. cubic) spline surfaces.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy.interpolate import interp2d, RectBivariateSpline

>>> x = np.arange(-5.01, 5.01, 0.25)
>>> y = np.arange(-5.01, 7.51, 0.25)
>>> xx, yy = np.meshgrid(x, y)
>>> z = np.sin(xx**2 + 2.*yy**2)
>>> f = interp2d(x, y, z, kind='cubic')
```

Example 2 (unknown):
```unknown
>>> z.size == len(x) * len(y)
True
```

Example 3 (unknown):
```unknown
>>> x.size, y.size
(41, 51)
```

Example 4 (python):
```python
>>> def plot(f, xnew, ynew):
...    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
...    znew = f(xnew, ynew)
...    ax1.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
...    im = ax2.imshow(znew)
...    plt.colorbar(im, ax=ax2)
...    plt.show()
...    return znew
...
>>> xnew = np.arange(-5.01, 5.01, 1e-2)
>>> ynew = np.arange(-5.01, 7.51, 1e-2)
>>> znew_i = plot(f, xnew, ynew)
```

---

## LAPACK functions for Cython#

**URL:** https://docs.scipy.org/doc/scipy/reference/linalg.cython_lapack.html

**Contents:**
- LAPACK functions for Cython#

Usable from Cython via:

This module provides Cython-level wrappers for all primary routines included in LAPACK 3.4.0 except for zcgesv since its interface is not consistent from LAPACK 3.4.0 to 3.6.0. It also provides some of the fixed-api auxiliary routines.

These wrappers do not check for alignment of arrays. Alignment should be checked before these wrappers are used.

Raw function pointers (Fortran-style pointer arguments):

**Examples:**

Example 1 (unknown):
```unknown
cimport scipy.linalg.cython_lapack
```

---

## RBFInterpolator#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html

**Contents:**
- RBFInterpolator#

Radial basis function interpolator in N ≥ 1 dimensions.

2-D array of data point coordinates.

N-D array of data values at y. The length of d along the first axis must be equal to the length of y. Unlike some interpolators, the interpolation axis cannot be changed.

If specified, the value of the interpolant at each evaluation point will be computed using only this many nearest data points. All the data points are used by default.

Smoothing parameter. The interpolant perfectly fits the data when this is set to 0. For large values, the interpolant approaches a least squares fit of a polynomial with the specified degree. Default is 0.

Type of RBF. This should be one of

‘thin_plate_spline’ : r**2 * log(r)

‘multiquadric’ : -sqrt(1 + r**2)

‘inverse_multiquadric’ : 1/sqrt(1 + r**2)

‘inverse_quadratic’ : 1/(1 + r**2)

‘gaussian’ : exp(-r**2)

Default is ‘thin_plate_spline’.

Shape parameter that scales the input to the RBF. If kernel is ‘linear’, ‘thin_plate_spline’, ‘cubic’, or ‘quintic’, this defaults to 1 and can be ignored because it has the same effect as scaling the smoothing parameter. Otherwise, this must be specified.

Degree of the added polynomial. For some RBFs the interpolant may not be well-posed if the polynomial degree is too small. Those RBFs and their corresponding minimum degrees are

‘thin_plate_spline’ : 1

The default value is the minimum degree for kernel or 0 if there is no minimum degree. Set this to -1 for no added polynomial.

Evaluate the interpolant at x.

An RBF is a scalar valued function in N-dimensional space whose value at \(x\) can be expressed in terms of \(r=||x - c||\), where \(c\) is the center of the RBF.

An RBF interpolant for the vector of data values \(d\), which are from locations \(y\), is a linear combination of RBFs centered at \(y\) plus a polynomial with a specified degree. The RBF interpolant is written as

where \(K(x, y)\) is a matrix of RBFs with centers at \(y\) evaluated at the points \(x\), and \(P(x)\) is a matrix of monomials, which span polynomials with the specified degree, evaluated at \(x\). The coefficients \(a\) and \(b\) are the solution to the linear equations

where \(\lambda\) is a non-negative smoothing parameter that controls how well we want to fit the data. The data are fit exactly when the smoothing parameter is 0.

The above system is uniquely solvable if the following requirements are met:

\(P(y)\) must have full column rank. \(P(y)\) always has full column rank when degree is -1 or 0. When degree is 1, \(P(y)\) has full column rank if the data point locations are not all collinear (N=2), coplanar (N=3), etc.

If kernel is ‘multiquadric’, ‘linear’, ‘thin_plate_spline’, ‘cubic’, or ‘quintic’, then degree must not be lower than the minimum value listed above.

If smoothing is 0, then each data point location must be distinct.

When using an RBF that is not scale invariant (‘multiquadric’, ‘inverse_multiquadric’, ‘inverse_quadratic’, or ‘gaussian’), an appropriate shape parameter must be chosen (e.g., through cross validation). Smaller values for the shape parameter correspond to wider RBFs. The problem can become ill-conditioned or singular when the shape parameter is too small.

The memory required to solve for the RBF interpolation coefficients increases quadratically with the number of data points, which can become impractical when interpolating more than about a thousand data points. To overcome memory limitations for large interpolation problems, the neighbors argument can be specified to compute an RBF interpolant for each evaluation point using only the nearest data points.

Added in version 1.7.0.

Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab. World Scientific Publishing Co.

http://amadeus.math.iit.edu/~fass/603_ch3.pdf

Wahba, G., 1990. Spline Models for Observational Data. SIAM.

http://pages.stat.wisc.edu/~wahba/stat860public/lect/lect8/lect8.pdf

Demonstrate interpolating scattered data to a grid in 2-D.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy.interpolate import RBFInterpolator
>>> from scipy.stats.qmc import Halton
```

Example 2 (unknown):
```unknown
>>> rng = np.random.default_rng()
>>> xobs = 2*Halton(2, seed=rng).random(100) - 1
>>> yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))
```

Example 3 (json):
```json
>>> xgrid = np.mgrid[-1:1:50j, -1:1:50j]
>>> xflat = xgrid.reshape(2, -1).T
>>> yflat = RBFInterpolator(xobs, yobs)(xflat)
>>> ygrid = yflat.reshape(50, 50)
```

Example 4 (unknown):
```unknown
>>> fig, ax = plt.subplots()
>>> ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
>>> p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
>>> fig.colorbar(p)
>>> plt.show()
```

---

## fiedler#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.fiedler.html

**Contents:**
- fiedler#

Returns a symmetric Fiedler matrix

Given an sequence of numbers a, Fiedler matrices have the structure F[i, j] = np.abs(a[i] - a[j]), and hence zero diagonals and nonnegative entries. A Fiedler matrix has a dominant positive eigenvalue and other eigenvalues are negative. Although not valid generally, for certain inputs, the inverse and the determinant can be derived explicitly as given in [1].

Coefficient array. N-dimensional arrays are treated as a batch: each slice along the last axis is a 1-D coefficient array.

Fiedler matrix. For batch input, each slice of shape (n, n) along the last two dimensions of the output corresponds with a slice of shape (n,) along the last dimension of the input.

Added in version 1.3.0.

J. Todd, “Basic Numerical Mathematics: Vol.2 : Numerical Algebra”, 1977, Birkhauser, DOI:10.1007/978-3-0348-7286-7

The explicit formulas for determinant and inverse seem to hold only for monotonically increasing/decreasing arrays. Note the tridiagonal structure and the corners.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import det, inv, fiedler
>>> a = [1, 4, 12, 45, 77]
>>> n = len(a)
>>> A = fiedler(a)
>>> A
array([[ 0,  3, 11, 44, 76],
       [ 3,  0,  8, 41, 73],
       [11,  8,  0, 33, 65],
       [44, 41, 33,  0, 32],
       [76, 73, 65, 32,  0]])
```

Example 2 (json):
```json
>>> Ai = inv(A)
>>> Ai[np.abs(Ai) < 1e-12] = 0.  # cleanup the numerical noise for display
>>> Ai
array([[-0.16008772,  0.16666667,  0.        ,  0.        ,  0.00657895],
       [ 0.16666667, -0.22916667,  0.0625    ,  0.        ,  0.        ],
       [ 0.        ,  0.0625    , -0.07765152,  0.01515152,  0.        ],
       [ 0.        ,  0.        ,  0.01515152, -0.03077652,  0.015625  ],
       [ 0.00657895,  0.        ,  0.        ,  0.015625  , -0.00904605]])
>>> det(A)
15409151.999999998
>>> (-1)**(n-1) * 2**(n-2) * np.diff(a).prod() * (a[-1] - a[0])
15409152
```

---

## scipy.special.airy#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.airy.html

**Contents:**
- scipy.special.airy#

Airy functions and their derivatives.

Real or complex argument.

Optional output arrays for the function values

Airy functions Ai and Bi, and their derivatives Aip and Bip.

exponentially scaled Airy functions.

The Airy functions Ai and Bi are two independent solutions of

For real z in [-10, 10], the computation is carried out by calling the Cephes [1] airy routine, which uses power series summation for small z and rational minimax approximations for large z.

Outside this range, the AMOS [2] zairy and zbiry routines are employed. They are computed using power series for \(|z| < 1\) and the following relations to modified Bessel functions for larger z (where \(t \equiv 2 z^{3/2}/3\)):

Cephes Mathematical Functions Library, http://www.netlib.org/cephes/

Donald E. Amos, “AMOS, A Portable Package for Bessel Functions of a Complex Argument and Nonnegative Order”, http://netlib.org/amos/

Compute the Airy functions on the interval [-15, 5].

Plot Ai(x) and Bi(x).

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import special
>>> x = np.linspace(-15, 5, 201)
>>> ai, aip, bi, bip = special.airy(x)
```

Example 2 (typescript):
```typescript
>>> import matplotlib.pyplot as plt
>>> plt.plot(x, ai, 'r', label='Ai(x)')
>>> plt.plot(x, bi, 'b--', label='Bi(x)')
>>> plt.ylim(-0.5, 1.0)
>>> plt.grid()
>>> plt.legend(loc='upper left')
>>> plt.show()
```

---

## generic_laplace#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_laplace.html

**Contents:**
- generic_laplace#

N-D Laplace filter using a provided second derivative function.

Callable with the following signature:

See extra_arguments, extra_keywords below.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

dict of extra keyword arguments to pass to passed function.

Sequence of extra positional arguments to pass to passed function.

The axes over which to apply the filter. If a mode tuple is provided, its length must match the number of axes.

Filtered array. Has the same shape as input.

**Examples:**

Example 1 (unknown):
```unknown
derivative2(input, axis, output, mode, cval,
            *extra_arguments, **extra_keywords)
```

---

## idct#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idct.html

**Contents:**
- idct#

Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

Type of the DCT (see Notes). Default type is 2.

Length of the transform. If n < x.shape[axis], x is truncated. If n > x.shape[axis], x is zero-padded. The default results in n = x.shape[axis].

Axis along which the idct is computed; the default is over the last axis (i.e., axis=-1).

Normalization mode (see Notes). Default is “backward”.

If True, the contents of x can be destroyed; the default is False.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

Whether to use the orthogonalized IDCT variant (see Notes). Defaults to True when norm="ortho" and False otherwise.

Added in version 1.8.0.

The transformed input array.

For a single dimension array x, idct(x, norm='ortho') is equal to MATLAB idct(x).

For type in {1, 2, 3}, norm="ortho" breaks the direct correspondence with the inverse direct Fourier transform. To recover it you must specify orthogonalize=False.

For norm="ortho" both the dct and idct are scaled by the same overall factor in both directions. By default, the transform is also orthogonalized which for types 1, 2 and 3 means the transform definition is modified to give orthogonality of the IDCT matrix (see dct for the full definitions).

‘The’ IDCT is the IDCT-II, which is the same as the normalized DCT-III.

The IDCT is equivalent to a normal DCT except for the normalization and type. DCT type 1 and 4 are their own inverse and DCTs 2 and 3 are each other’s inverses.

The Type 1 DCT is equivalent to the DFT for real, even-symmetrical inputs. The output is also real and even-symmetrical. Half of the IFFT input is used to generate half of the IFFT output:

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.fft import ifft, idct
>>> import numpy as np
>>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
array([  4.,   3.,   5.,  10.,   5.,   3.])
>>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1)
array([  4.,   3.,   5.,  10.])
```

---

## Thread Safety in SciPy#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/thread_safety.html

**Contents:**
- Thread Safety in SciPy#
- Free-threaded Python#

SciPy supports use in a multithreaded context via the threading module in the standard library. Many SciPy operations release the GIL, as does NumPy (and a lot of SciPy functionality is implemented as calls to NumPy functions) - so unlike many situations in Python, it is possible to improve parallel performance by exploiting multithreaded parallelism in Python.

The easiest performance gains happen when each worker thread owns its own array or set of array objects, with no data directly shared between threads. Threads that spend most of their time in low-level code will typically run in parallel.

It is possible to share NumPy arrays between threads, but extreme care must be taken to avoid creating thread safety issues when mutating arrays that are shared between multiple threads - please see the NumPy documentation on thread safety for more details. SciPy functions will not mutate arrays that the user passes in, unless a function explicitly documents that it will do so (which is rare). Hence calling SciPy functions in a threaded fashion on the same NumPy array is safe.

While most of SciPy consists of functions, more care has to be taken with classes and data structures.

Classes that have state, such as some of the integration and interpolation objects in scipy.integrate and scipy.interpolate, are typically robust against being called in parallel. They either accept parallel calls or raise an informative error. For example, scipy.integrate.ode may raise an IntegratorConcurrencyError for integration methods that do not support parallel execution.

SciPy offers a couple of data structures, namely sparse arrays and matrices in scipy.sparse, and k-D trees in scipy.spatial. These data structures are currently not thread-safe. Please avoid in particular operations that mutate a data structure, like using item or slice assignment on sparse arrays, while the data is shared across multiple threads. That may result in data corruption, crashes, or other unwanted behavior.

Note that operations that do not release the GIL will see no performance gains from use of the threading module, and instead might be better served with multiprocessing.

Added in version 1.15.0.

Starting with SciPy 1.15.0 and CPython 3.13, SciPy has experimental support for Python runtimes with the GIL disabled on all platforms. See https://py-free-threading.github.io for more information about installing and using free-threaded Python.

Because free-threaded Python does not have a global interpreter lock (GIL) to serialize access to Python objects, there are more opportunities for threads to mutate shared state and create thread safety issues. All SciPy functionality is tested for usage from parallel threads, however we expect there to be issues that are as yet undiscovered - if you run into a problem, please check the GitHub issues with the free-threading label and open a new issue if one does not exist yet for the function that is misbehaving.

---

## whosmat#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.whosmat.html

**Contents:**
- whosmat#

List variables inside a MATLAB file.

Name of the mat file (do not need .mat extension if appendmat==True) Can also pass open file-like object.

True to append the .mat extension to the end of the given filename, if not already present. Default is True.

None by default, implying byte order guessed from mat file. Otherwise can be one of (‘native’, ‘=’, ‘little’, ‘<’, ‘BIG’, ‘>’).

If True, return arrays in same dtype as would be loaded into MATLAB (instead of the dtype with which they are saved).

Whether to squeeze unit matrix dimensions or not.

Whether to convert char arrays to string arrays.

Returns matrices as would be loaded by MATLAB (implies squeeze_me=False, chars_as_strings=False, mat_dtype=True, struct_as_record=True).

Whether to load MATLAB structs as NumPy record arrays, or as old-style NumPy arrays with dtype=object. Setting this flag to False replicates the behavior of SciPy version 0.7.x (returning numpy object arrays). The default setting is True, because it allows easier round-trip load and save of MATLAB files.

A list of tuples, where each tuple holds the matrix name (a string), its shape (tuple of ints), and its data class (a string). Possible data classes are: int8, uint8, int16, uint16, int32, uint32, int64, uint64, single, double, cell, struct, object, char, sparse, function, opaque, logical, unknown.

v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

You will need an HDF5 python library to read matlab 7.3 format mat files (e.g. h5py). Because SciPy does not supply one, we do not implement the HDF5 / 7.3 interface here.

Added in version 0.12.0.

Create some arrays, and use savemat to write them to a BytesIO instance.

Use whosmat to inspect f. Each tuple in the output list gives the name, shape and data type of the array in f.

**Examples:**

Example 1 (python):
```python
>>> from io import BytesIO
>>> import numpy as np
>>> from scipy.io import savemat, whosmat
```

Example 2 (unknown):
```unknown
>>> a = np.array([[10, 20, 30], [11, 21, 31]], dtype=np.int32)
>>> b = np.geomspace(1, 10, 5)
>>> f = BytesIO()
>>> savemat(f, {'a': a, 'b': b})
```

Example 3 (json):
```json
>>> whosmat(f)
[('a', (2, 3), 'int32'), ('b', (1, 5), 'double')]
```

---

## CloughTocher2DInterpolator#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html

**Contents:**
- CloughTocher2DInterpolator#

Piecewise cubic, C1 smooth, curvature-minimizing interpolator in N=2 dimensions.

Added in version 0.9.

2-D array of data point coordinates, or a precomputed Delaunay triangulation.

N-D array of data values at points. The length of values along the first axis must be equal to the length of points. Unlike some interpolators, the interpolation axis cannot be changed.

Value used to fill in for requested points outside of the convex hull of the input points. If not provided, then the default is nan.

Absolute/relative tolerance for gradient estimation.

Maximum number of iterations in gradient estimation.

Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.

Evaluate interpolator at given points.

Interpolate unstructured D-D data.

Piecewise linear interpolator in N > 1 dimensions.

Nearest-neighbor interpolator in N > 1 dimensions.

Interpolation on a regular grid or rectilinear grid.

Interpolator on a regular or rectilinear grid in arbitrary dimensions (interpn wraps this class).

The interpolant is constructed by triangulating the input data with Qhull [1], and constructing a piecewise cubic interpolating Bezier polynomial on each triangle, using a Clough-Tocher scheme [CT]. The interpolant is guaranteed to be continuously differentiable.

The gradients of the interpolant are chosen so that the curvature of the interpolating surface is approximatively minimized. The gradients necessary for this are estimated using the global algorithm described in [Nielson83] and [Renka84].

For data on a regular grid use interpn instead.

http://www.qhull.org/

See, for example, P. Alfeld, ‘’A trivariate Clough-Tocher scheme for tetrahedral data’’. Computer Aided Geometric Design, 1, 169 (1984); G. Farin, ‘’Triangular Bernstein-Bezier patches’’. Computer Aided Geometric Design, 3, 83 (1986).

G. Nielson, ‘’A method for interpolating scattered data based upon a minimum norm network’’. Math. Comp., 40, 253 (1983).

R. J. Renka and A. K. Cline. ‘’A Triangle-based C1 interpolation method.’’, Rocky Mountain J. Math., 14, 223 (1984).

We can interpolate values on a 2D plane:

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.interpolate import CloughTocher2DInterpolator
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> x = rng.random(10) - 0.5
>>> y = rng.random(10) - 0.5
>>> z = np.hypot(x, y)
>>> X = np.linspace(min(x), max(x))
>>> Y = np.linspace(min(y), max(y))
>>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
>>> interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
>>> Z = interp(X, Y)
>>> plt.pcolormesh(X, Y, Z, shading='auto')
>>> plt.plot(x, y, "ok", label="input point")
>>> plt.legend(loc="upper right")
>>> plt.colorbar()
>>> plt.axis("equal")
>>> plt.show()
```

---

## generic_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html

**Contents:**
- generic_filter#

Calculate a multidimensional filter using the given function.

At each element the provided function is called. The input values within the filter footprint at that element are passed to the function as a 1-D array of double values.

Function to apply at each element.

See footprint, below. Ignored if footprint is given.

Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function. footprint is a boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)). We adjust size to the number of dimensions of the input array, so that, if the input array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When footprint is given, size is ignored.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

Sequence of extra positional arguments to pass to passed function.

dict of extra keyword arguments to pass to passed function.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for size or origin must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

Filtered array. Has the same shape as input.

similar functionality, but optimized for vectorized callables

This function is ideal for use with instances of scipy.LowLevelCallable; for vectorized, pure-Python callables, consider vectorized_filter for improved performance.

Low-level callback functions must have one of the following signatures:

The calling function iterates over the elements of the input and output arrays, calling the callback function at each element. The elements within the footprint of the filter at the current element are passed through the buffer parameter, and the number of elements within the footprint through filter_size. The calculated value is returned in return_value. user_data is the data pointer provided to scipy.LowLevelCallable as-is.

The callback function must return an integer error status that is zero if something went wrong and one otherwise. If an error occurs, you should normally set the python error status with an informative message before returning, otherwise a default error message is set by the calling function.

In addition, some other low-level function pointer specifications are accepted, but these are for backward compatibility only and should not be used in new code.

Import the necessary modules and load the example image used for filtering.

Compute a maximum filter with kernel size 5 by passing a simple NumPy aggregation function as argument to function.

While a maximum filter could also directly be obtained using maximum_filter, generic_filter allows generic Python function or scipy.LowLevelCallable to be used as a filter. Here, we compute the range between maximum and minimum value as an example for a kernel size of 5.

Plot the original and filtered images.

**Examples:**

Example 1 (unknown):
```unknown
int callback(double *buffer, npy_intp filter_size,
             double *return_value, void *user_data)
int callback(double *buffer, intptr_t filter_size,
             double *return_value, void *user_data)
```

Example 2 (python):
```python
>>> import numpy as np
>>> from scipy import datasets
>>> from scipy.ndimage import zoom, generic_filter
>>> import matplotlib.pyplot as plt
>>> ascent = zoom(datasets.ascent(), 0.5)
```

Example 3 (unknown):
```unknown
>>> maximum_filter_result = generic_filter(ascent, np.amax, [5, 5])
```

Example 4 (python):
```python
>>> def custom_filter(image):
...     return np.amax(image) - np.amin(image)
>>> custom_filter_result = generic_filter(ascent, custom_filter, [5, 5])
```

---

## map_coordinates#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html

**Contents:**
- map_coordinates#

Map the input array to new coordinates by interpolation.

The array of coordinates is used to find, for each point in the output, the corresponding coordinates in the input. The value of the input at those coordinates is determined by spline interpolation of the requested order.

The shape of the output is derived from that of the coordinate array by dropping the first axis. The values of the array along the first axis are the coordinates in the input array at which the output value is found.

The coordinates at which input is evaluated.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘constant’. Behavior for each valid value is as follows (see additional plots and details on boundary modes):

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

This is a synonym for ‘reflect’.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. No interpolation is performed beyond the edges of the input.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. Interpolation occurs for samples outside the input’s extent as well.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

The input is extended by wrapping around to the opposite edge, but in a way such that the last point and initial point exactly overlap. In this case it is not well defined which sample will be chosen at the point of overlap.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Determines if the input array is prefiltered with spline_filter before interpolation. The default is True, which will create a temporary float64 array of filtered values if order > 1. If setting this to False, the output will be slightly blurred if order > 1, unless the input is prefiltered, i.e. it is the result of calling spline_filter on the original input.

The result of transforming the input. The shape of the output is derived from that of coordinates by dropping the first axis.

For complex-valued input, this function maps the real and imaginary components independently.

Added in version 1.6.0: Complex-valued support added.

Above, the interpolated value of a[0.5, 0.5] gives output[0], while a[2, 1] is output[1].

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage
>>> import numpy as np
>>> a = np.arange(12.).reshape((4, 3))
>>> a
array([[  0.,   1.,   2.],
       [  3.,   4.,   5.],
       [  6.,   7.,   8.],
       [  9.,  10.,  11.]])
>>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)
array([ 2.,  7.])
```

Example 2 (unknown):
```unknown
>>> inds = np.array([[0.5, 2], [0.5, 4]])
>>> ndimage.map_coordinates(a, inds, order=1, cval=-33.3)
array([  2. , -33.3])
>>> ndimage.map_coordinates(a, inds, order=1, mode='nearest')
array([ 2.,  8.])
>>> ndimage.map_coordinates(a, inds, order=1, cval=0, output=bool)
array([ True, False], dtype=bool)
```

---

## fft#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html

**Contents:**
- fft#

Compute the 1-D discrete Fourier Transform.

This function computes the 1-D n-point discrete Fourier Transform (DFT) with the efficient Fast Fourier Transform (FFT) algorithm [1].

Input array, can be complex.

Length of the transformed axis of the output. If n is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros. If n is not given, the length of the input along the axis specified by axis is used.

Axis over which to compute the FFT. If not given, the last axis is used.

Normalization mode. Default is “backward”, meaning no normalization on the forward transforms and scaling by 1/n on the ifft. “forward” instead applies the 1/n factor on the forward transform. For norm="ortho", both directions are scaled by 1/sqrt(n).

Added in version 1.6.0: norm={"forward", "backward"} options were added

If True, the contents of x can be destroyed; the default is False. See the notes below for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See below for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified.

if axes is larger than the last axis of x.

The N-D FFT of real input.

Frequency bins for given FFT parameters.

Size to pad input to for most efficient transforms

FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform (DFT) can be calculated efficiently, by using symmetries in the calculated terms. The symmetry is highest when n is a power of 2, and the transform is therefore most efficient for these sizes. For poorly factorizable sizes, scipy.fft uses Bluestein’s algorithm [2] and so is never worse than O(n log n). Further performance improvements may be seen by zero-padding the input using next_fast_len.

If x is a 1d array, then the fft is equivalent to

The frequency term f=k/n is found at y[k]. At y[n/2] we reach the Nyquist frequency and wrap around to the negative-frequency terms. So, for an 8-point transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1]. To rearrange the fft output so that the zero-frequency component is centered, like [-4, -3, -2, -1, 0, 1, 2, 3], use fftshift.

Transforms can be done in single, double, or extended precision (long double) floating point. Half precision inputs will be converted to single precision and non-floating-point inputs will be converted to double precision.

If the data type of x is real, a “real FFT” algorithm is automatically used, which roughly halves the computation time. To increase efficiency a little further, use rfft, which does the same calculation, but only outputs half of the symmetrical spectrum. If the data are both real and symmetrical, the dct can again double the efficiency, by generating half of the spectrum from half of the signal.

When overwrite_x=True is specified, the memory referenced by x may be used by the implementation in any way. This may include reusing the memory for the result, but this is in no way guaranteed. You should not rely on the contents of x after the transform as this may change in future without warning.

The workers argument specifies the maximum number of parallel jobs to split the FFT computation into. This will execute independent 1-D FFTs within x. So, x must be at least 2-D and the non-transformed axes must be large enough to split into chunks. If x is too small, fewer jobs may be used than requested.

Cooley, James W., and John W. Tukey, 1965, “An algorithm for the machine calculation of complex Fourier series,” Math. Comput. 19: 297-301.

Bluestein, L., 1970, “A linear filtering approach to the computation of discrete Fourier transform”. IEEE Transactions on Audio and Electroacoustics. 18 (4): 451-455.

In this example, real input has an FFT which is Hermitian, i.e., symmetric in the real part and anti-symmetric in the imaginary part:

**Examples:**

Example 1 (unknown):
```unknown
y[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n)/n))
```

Example 2 (typescript):
```typescript
>>> import scipy.fft
>>> import numpy as np
>>> scipy.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
        2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
       -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
        1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])
```

Example 3 (sql):
```sql
>>> from scipy.fft import fft, fftfreq, fftshift
>>> import matplotlib.pyplot as plt
>>> t = np.arange(256)
>>> sp = fftshift(fft(np.sin(t)))
>>> freq = fftshift(fftfreq(t.shape[-1]))
>>> plt.plot(freq, sp.real, freq, sp.imag)
[<matplotlib.lines.Line2D object at 0x...>,
 <matplotlib.lines.Line2D object at 0x...>]
>>> plt.show()
```

---

## lu_factor#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html

**Contents:**
- lu_factor#

Compute pivoted LU decomposition of a matrix.

The decomposition is:

where P is a permutation matrix, L lower triangular with unit diagonal elements, and U upper triangular.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Whether to overwrite data in A (may increase performance)

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Matrix containing U in its upper triangle, and L in its lower triangle. The unit diagonal elements of L are not stored.

Pivot indices representing the permutation matrix P: row i of matrix was interchanged with row piv[i]. Of shape (K,), with K = min(M, N).

gives lu factorization in more user-friendly format

solve an equation system using the LU factorization of a matrix

This is a wrapper to the *GETRF routines from LAPACK. Unlike lu, it outputs the L and U factors into a single array and returns pivot indices instead of a permutation matrix.

While the underlying *GETRF routines return 1-based pivot indices, the piv array returned by lu_factor contains 0-based indices.

Convert LAPACK’s piv array to NumPy index and test the permutation

The P matrix in P L U is defined by the inverse permutation and can be recovered using argsort:

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import lu_factor
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> lu, piv = lu_factor(A)
>>> piv
array([2, 2, 3, 3], dtype=int32)
```

Example 2 (python):
```python
>>> def pivot_to_permutation(piv):
...     perm = np.arange(len(piv))
...     for i in range(len(piv)):
...         perm[i], perm[piv[i]] = perm[piv[i]], perm[i]
...     return perm
...
>>> p_inv = pivot_to_permutation(piv)
>>> p_inv
array([2, 0, 3, 1])
>>> L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)
>>> np.allclose(A[p_inv] - L @ U, np.zeros((4, 4)))
True
```

Example 3 (unknown):
```unknown
>>> p = np.argsort(p_inv)
>>> p
array([1, 3, 0, 2])
>>> np.allclose(A - L[p] @ U, np.zeros((4, 4)))
True
```

Example 4 (unknown):
```unknown
>>> P = np.eye(4)[p]
>>> np.allclose(A - P @ L @ U, np.zeros((4, 4)))
True
```

---

## The main SciPy namespace#

**URL:** https://docs.scipy.org/doc/scipy/reference/main_namespace.html

**Contents:**
- The main SciPy namespace#
- Submodules#

The main scipy namespace has very few objects in it by design. Only show generical functionality related to testing, build info and versioning, and one class (LowLevelCallable) that didn’t fit in one of the submodules, are present:

LowLevelCallable(function[, user_data, ...])

Low-level callback function.

Show libraries and system information on which SciPy was built and is being used

Run tests for this namespace

The one public attribute is:

Clustering functionality

Physical and mathematical constants and units

Discrete Fourier and related transforms

Discrete Fourier transforms (legacy)

Numerical integration and ODEs

Scientific data format reading and writing

Linear algebra functionality

Utility routines (deprecated)

N-dimensional image processing and interpolation

Orthogonal distance regression

Numerical optimization

Sparse arrays, linear algebra and graph algorithms

Spatial data structures and algorithms

Statistical functions

---

## Scattered data interpolation (griddata)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_unstructured.html

**Contents:**
- Scattered data interpolation (griddata)#
- Using radial basis functions for smoothing/interpolation#
- 1-D Example#
- 2-D Example#

Suppose you have multidimensional data, for instance, for an underlying function \(f(x, y)\) you only know the values at points (x[i], y[i]) that do not form a regular grid.

Suppose we want to interpolate the 2-D function

on a grid in [0, 1]x[0, 1]

but we only know its values at 1000 data points:

This can be done with griddata – below, we try out all of the interpolation methods:

One can see that the exact result is reproduced by all of the methods to some degree, but for this smooth function the piecewise cubic interpolant gives the best results (black dots show the data being interpolated):

For each interpolation method, this function delegates to a corresponding class object — these classes can be used directly as well — NearestNDInterpolator, LinearNDInterpolator and CloughTocher2DInterpolator for piecewise cubic interpolation in 2D.

All these interpolation methods rely on triangulation of the data using the QHull library wrapped in scipy.spatial.

griddata is based on triangulation, hence is appropriate for unstructured, scattered data. If your data is on a full grid, the griddata function — despite its name — is not the right tool. Use RegularGridInterpolator instead.

If the input data is such that input dimensions have incommensurate units and differ by many orders of magnitude, the interpolant may have numerical artifacts. Consider rescaling the data before interpolating or use the rescale=True keyword argument to griddata.

Radial basis functions can be used for smoothing/interpolating scattered data in N dimensions, but should be used with caution for extrapolation outside of the observed data range.

This example compares the usage of the RBFInterpolator and UnivariateSpline classes from the scipy.interpolate module.

This example shows how to interpolate scattered 2-D data:

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> def func(x, y):
...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
```

Example 2 (unknown):
```unknown
>>> grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100),
...                              np.linspace(0, 1, 200), indexing='ij')
```

Example 3 (unknown):
```unknown
>>> rng = np.random.default_rng()
>>> points = rng.random((1000, 2))
>>> values = func(points[:,0], points[:,1])
```

Example 4 (sql):
```sql
>>> from scipy.interpolate import griddata
>>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
>>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
>>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
```

---

## Input and output (scipy.io)#

**URL:** https://docs.scipy.org/doc/scipy/reference/io.html

**Contents:**
- Input and output (scipy.io)#
- MATLAB® files#
- IDL® files#
- Matrix Market files#
- Unformatted Fortran files#
- Netcdf#
- Harwell-Boeing files#
- Wav sound files (scipy.io.wavfile)#
- Arff files (scipy.io.arff)#

SciPy has many modules, classes, and functions available to read data from and write data to a variety of file formats.

loadmat(file_name[, mdict, appendmat, spmatrix])

savemat(file_name, mdict[, appendmat, ...])

Save a dictionary of names and arrays into a MATLAB-style .mat file.

whosmat(file_name[, appendmat])

List variables inside a MATLAB file.

For low-level MATLAB reading and writing utilities, see scipy.io.matlab.

readsav(file_name[, idict, python_dict, ...])

Read an IDL .sav file.

Return size and storage parameters from Matrix Market file-like 'source'.

mmread(source, *[, spmatrix])

Reads the contents of a Matrix Market file-like 'source' into a matrix.

mmwrite(target, a[, comment, field, ...])

Writes the sparse or dense array a to Matrix Market file-like target.

FortranFile(filename[, mode, header_dtype])

A file object for unformatted sequential files from Fortran code.

Indicates that the file ended properly.

FortranFormattingError

Indicates that the file ended mid-record.

netcdf_file(filename[, mode, mmap, version, ...])

A file object for NetCDF data.

netcdf_variable(data, typecode, size, shape, ...)

A data object for netcdf files.

hb_read(path_or_open_file, *[, spmatrix])

hb_write(path_or_open_file, m[, hb_info])

Write HB-format file.

read(filename[, mmap])

write(filename, rate, data)

Write a NumPy array as a WAV file.

Small container to keep useful information on a ARFF dataset.

---

## diagsvd#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.diagsvd.html

**Contents:**
- diagsvd#

Construct the sigma matrix in SVD from singular values and size M, N.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Size of the matrix whose singular values are s.

Size of the matrix whose singular values are s.

The S-matrix in the singular value decomposition

Singular value decomposition of a matrix

Compute singular values of a matrix.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import diagsvd
>>> vals = np.array([1, 2, 3])  # The array representing the computed svd
>>> diagsvd(vals, 3, 4)
array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0]])
>>> diagsvd(vals, 4, 3)
array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3],
       [0, 0, 0]])
```

---

## CubicSpline#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html

**Contents:**
- CubicSpline#

Piecewise cubic interpolator to fit values (C2 smooth).

Interpolate data with a piecewise cubic polynomial which is twice continuously differentiable [1]. The result is represented as a PPoly instance with breakpoints matching the given data.

1-D array containing values of the independent variable. Values must be real, finite and in strictly increasing order.

Array containing values of the dependent variable. It can have arbitrary number of dimensions, but the length along axis (see below) must match the length of x. Values must be finite.

Axis along which y is assumed to be varying. Meaning that for x[i] the corresponding values are np.take(y, i, axis=axis). Default is 0.

Boundary condition type. Two additional equations, given by the boundary conditions, are required to determine all coefficients of polynomials on each segment [2].

If bc_type is a string, then the specified condition will be applied at both ends of a spline. Available conditions are:

‘not-a-knot’ (default): The first and second segment at a curve end are the same polynomial. It is a good default when there is no information on boundary conditions.

‘periodic’: The interpolated functions is assumed to be periodic of period x[-1] - x[0]. The first and last value of y must be identical: y[0] == y[-1]. This boundary condition will result in y'[0] == y'[-1] and y''[0] == y''[-1].

‘clamped’: The first derivative at curves ends are zero. Assuming a 1D y, bc_type=((1, 0.0), (1, 0.0)) is the same condition.

‘natural’: The second derivative at curve ends are zero. Assuming a 1D y, bc_type=((2, 0.0), (2, 0.0)) is the same condition.

If bc_type is a 2-tuple, the first and the second value will be applied at the curve start and end respectively. The tuple values can be one of the previously mentioned strings (except ‘periodic’) or a tuple (order, deriv_values) allowing to specify arbitrary derivatives at curve ends:

order: the derivative order, 1 or 2.

deriv_value: array_like containing derivative values, shape must be the same as y, excluding axis dimension. For example, if y is 1-D, then deriv_value must be a scalar. If y is 3-D with the shape (n0, n1, n2) and axis=2, then deriv_value must be 2-D and have the shape (n0, n1).

If bool, determines whether to extrapolate to out-of-bounds points based on first and last intervals, or to return NaNs. If ‘periodic’, periodic extrapolation is used. If None (default), extrapolate is set to ‘periodic’ for bc_type='periodic' and to True otherwise.

Breakpoints. The same x which was passed to the constructor.

Coefficients of the polynomials on each segment. The trailing dimensions match the dimensions of y, excluding axis. For example, if y is 1-d, then c[k, i] is a coefficient for (x-x[i])**(3-k) on the segment between x[i] and x[i+1].

Interpolation axis. The same axis which was passed to the constructor.

__call__(x[, nu, extrapolate])

Evaluate the piecewise polynomial or its derivative.

Construct a new piecewise polynomial representing the derivative.

Construct a new piecewise polynomial representing the antiderivative.

integrate(a, b[, extrapolate])

Compute a definite integral over a piecewise polynomial.

solve([y, discontinuity, extrapolate])

Find real solutions of the equation pp(x) == y.

roots([discontinuity, extrapolate])

Find real roots of the piecewise polynomial.

Akima 1D interpolator.

PCHIP 1-D monotonic cubic interpolator.

Piecewise polynomial in terms of coefficients and breakpoints.

Parameters bc_type and extrapolate work independently, i.e. the former controls only construction of a spline, and the latter only evaluation.

When a boundary condition is ‘not-a-knot’ and n = 2, it is replaced by a condition that the first derivative is equal to the linear interpolant slope. When both boundary conditions are ‘not-a-knot’ and n = 3, the solution is sought as a parabola passing through given points.

When ‘not-a-knot’ boundary conditions is applied to both ends, the resulting spline will be the same as returned by splrep (with s=0) and InterpolatedUnivariateSpline, but these two methods use a representation in B-spline basis.

Added in version 0.18.0.

Cubic Spline Interpolation on Wikiversity.

Carl de Boor, “A Practical Guide to Splines”, Springer-Verlag, 1978.

In this example the cubic spline is used to interpolate a sampled sinusoid. You can see that the spline continuity property holds for the first and second derivatives and violates only for the third derivative.

In the second example, the unit circle is interpolated with a spline. A periodic boundary condition is used. You can see that the first derivative values, ds/dx=0, ds/dy=1 at the periodic point (1, 0) are correctly computed. Note that a circle cannot be exactly represented by a cubic spline. To increase precision, more breakpoints would be required.

The third example is the interpolation of a polynomial y = x**3 on the interval 0 <= x<= 1. A cubic spline can represent this function exactly. To achieve that we need to specify values and first derivatives at endpoints of the interval. Note that y’ = 3 * x**2 and thus y’(0) = 0 and y’(1) = 3.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.interpolate import CubicSpline
>>> import matplotlib.pyplot as plt
>>> x = np.arange(10)
>>> y = np.sin(x)
>>> cs = CubicSpline(x, y)
>>> xs = np.arange(-0.5, 9.6, 0.1)
>>> fig, ax = plt.subplots(figsize=(6.5, 4))
>>> ax.plot(x, y, 'o', label='data')
>>> ax.plot(xs, np.sin(xs), label='true')
>>> ax.plot(xs, cs(xs), label="S")
>>> ax.plot(xs, cs(xs, 1), label="S'")
>>> ax.plot(xs, cs(xs, 2), label="S''")
>>> ax.plot(xs, cs(xs, 3), label="S'''")
>>> ax.set_xlim(-0.5, 9.5)
>>> ax.legend(loc='lower left', ncol=2)
>>> plt.show()
```

Example 2 (unknown):
```unknown
>>> theta = 2 * np.pi * np.linspace(0, 1, 5)
>>> y = np.c_[np.cos(theta), np.sin(theta)]
>>> cs = CubicSpline(theta, y, bc_type='periodic')
>>> print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
ds/dx=0.0 ds/dy=1.0
>>> xs = 2 * np.pi * np.linspace(0, 1, 100)
>>> fig, ax = plt.subplots(figsize=(6.5, 4))
>>> ax.plot(y[:, 0], y[:, 1], 'o', label='data')
>>> ax.plot(np.cos(xs), np.sin(xs), label='true')
>>> ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
>>> ax.axes.set_aspect('equal')
>>> ax.legend(loc='center')
>>> plt.show()
```

Example 3 (unknown):
```unknown
>>> cs = CubicSpline([0, 1], [0, 1], bc_type=((1, 0), (1, 3)))
>>> x = np.linspace(0, 1)
>>> np.allclose(x**3, cs(x))
True
```

---

## dct#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html

**Contents:**
- dct#

Return the Discrete Cosine Transform of arbitrary type sequence x.

Type of the DCT (see Notes). Default type is 2.

Length of the transform. If n < x.shape[axis], x is truncated. If n > x.shape[axis], x is zero-padded. The default results in n = x.shape[axis].

Axis along which the dct is computed; the default is over the last axis (i.e., axis=-1).

Normalization mode (see Notes). Default is “backward”.

If True, the contents of x can be destroyed; the default is False.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

Whether to use the orthogonalized DCT variant (see Notes). Defaults to True when norm="ortho" and False otherwise.

Added in version 1.8.0.

The transformed input array.

For a single dimension array x, dct(x, norm='ortho') is equal to MATLAB dct(x).

For type in {1, 2, 3}, norm="ortho" breaks the direct correspondence with the direct Fourier transform. To recover it you must specify orthogonalize=False.

For norm="ortho" both the dct and idct are scaled by the same overall factor in both directions. By default, the transform is also orthogonalized which for types 1, 2 and 3 means the transform definition is modified to give orthogonality of the DCT matrix (see below).

For norm="backward", there is no scaling on dct and the idct is scaled by 1/N where N is the “logical” size of the DCT. For norm="forward" the 1/N normalization is applied to the forward dct instead and the idct is unnormalized.

There are, theoretically, 8 types of the DCT, only the first 4 types are implemented in SciPy.’The’ DCT generally refers to DCT type 2, and ‘the’ Inverse DCT generally refers to DCT type 3.

There are several definitions of the DCT-I; we use the following (for norm="backward")

If orthogonalize=True, x[0] and x[N-1] are multiplied by a scaling factor of \(\sqrt{2}\), and y[0] and y[N-1] are divided by \(\sqrt{2}\). When combined with norm="ortho", this makes the corresponding matrix of coefficients orthonormal (O @ O.T = np.eye(N)).

The DCT-I is only supported for input size > 1.

There are several definitions of the DCT-II; we use the following (for norm="backward")

If orthogonalize=True, y[0] is divided by \(\sqrt{2}\) which, when combined with norm="ortho", makes the corresponding matrix of coefficients orthonormal (O @ O.T = np.eye(N)).

There are several definitions, we use the following (for norm="backward")

If orthogonalize=True, x[0] terms are multiplied by \(\sqrt{2}\) which, when combined with norm="ortho", makes the corresponding matrix of coefficients orthonormal (O @ O.T = np.eye(N)).

The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up to a factor 2N. The orthonormalized DCT-III is exactly the inverse of the orthonormalized DCT-II.

There are several definitions of the DCT-IV; we use the following (for norm="backward")

orthogonalize has no effect here, as the DCT-IV matrix is already orthogonal up to a scale factor of 2N.

‘A Fast Cosine Transform in One and Two Dimensions’, by J. Makhoul, IEEE Transactions on acoustics, speech and signal processing vol. 28(1), pp. 27-34, DOI:10.1109/TASSP.1980.1163351 (1980).

Wikipedia, “Discrete cosine transform”, https://en.wikipedia.org/wiki/Discrete_cosine_transform

The Type 1 DCT is equivalent to the FFT (though faster) for real, even-symmetrical inputs. The output is also real and even-symmetrical. Half of the FFT input is used to generate half of the FFT output:

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.fft import fft, dct
>>> import numpy as np
>>> fft(np.array([4., 3., 5., 10., 5., 3.])).real
array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])
>>> dct(np.array([4., 3., 5., 10.]), 1)
array([ 30.,  -8.,   6.,  -2.])
```

---

## Hierarchical clustering (scipy.cluster.hierarchy)#

**URL:** https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

**Contents:**
- Hierarchical clustering (scipy.cluster.hierarchy)#

These functions cut hierarchical clusterings into flat clusterings or find the roots of the forest formed by a cut by providing the flat cluster ids of each observation.

fcluster(Z, t[, criterion, depth, R, monocrit])

Form flat clusters from the hierarchical clustering defined by the given linkage matrix.

fclusterdata(X, t[, criterion, metric, ...])

Cluster observation data using a given metric.

Return the root nodes in a hierarchical clustering.

These are routines for agglomerative clustering.

linkage(y[, method, metric, optimal_ordering])

Perform hierarchical/agglomerative clustering.

Perform single/min/nearest linkage on the condensed distance matrix y.

Perform complete/max/farthest point linkage on a condensed distance matrix.

Perform average/UPGMA linkage on a condensed distance matrix.

Perform weighted/WPGMA linkage on the condensed distance matrix.

Perform centroid/UPGMC linkage.

Perform median/WPGMC linkage.

Perform Ward's linkage on a condensed distance matrix.

These routines compute statistics on hierarchies.

Calculate the cophenetic distances between each observation in the hierarchical clustering defined by the linkage Z.

Convert a linkage matrix generated by MATLAB(TM) to a new linkage matrix compatible with this module.

Calculate inconsistency statistics on a linkage matrix.

Return the maximum inconsistency coefficient for each non-singleton cluster and its children.

Return the maximum distance between any non-singleton cluster.

Return the maximum statistic for each non-singleton cluster and its children.

Convert a linkage matrix to a MATLAB(TM) compatible one.

Routines for visualizing flat clusters.

dendrogram(Z[, p, truncate_mode, ...])

Plot the hierarchical clustering as a dendrogram.

These are data structures and routines for representing hierarchies as tree objects.

ClusterNode(id[, left, right, dist, count])

A tree node class for representing a cluster.

Return a list of leaf node ids.

Convert a linkage matrix into an easy-to-use tree object.

cut_tree(Z[, n_clusters, height])

Given a linkage matrix Z, return the cut tree.

optimal_leaf_ordering(Z, y[, metric])

Given a linkage matrix Z and distance, reorder the cut tree.

These are predicates for checking the validity of linkage and inconsistency matrices as well as for checking isomorphism of two flat cluster assignments.

is_valid_im(R[, warning, throw, name])

Return True if the inconsistency matrix passed is valid.

is_valid_linkage(Z[, warning, throw, name])

Check the validity of a linkage matrix.

is_isomorphic(T1, T2)

Determine if two different cluster assignments are equivalent.

Return True if the linkage passed is monotonic.

Check for correspondence between linkage and condensed distance matrices.

Return the number of original observations of the linkage matrix passed.

Utility routines for plotting:

set_link_color_palette(palette)

Set list of matplotlib color codes for use by dendrogram.

DisjointSet([elements])

Disjoint set data structure for incremental connectivity queries.

---

## nquad#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.nquad.html

**Contents:**
- nquad#

Integration over multiple variables.

Wraps quad to enable integration over multiple variables. Various options allow improved integration of discontinuous functions, as well as the use of weighted integration, and generally finer control of the integration process.

The function to be integrated. Has arguments of x0, ... xn, t0, ... tm, where integration is carried out over x0, ... xn, which must be floats. Where t0, ... tm are extra arguments passed in args. Function signature should be func(x0, x1, ..., xn, t0, t1, ..., tm). Integration is carried out in order. That is, integration over x0 is the innermost integral, and xn is the outermost.

If the user desires improved integration performance, then f may be a scipy.LowLevelCallable with one of the signatures:

where n is the number of variables and args. The xx array contains the coordinates and extra arguments. user_data is the data contained in the scipy.LowLevelCallable.

Each element of ranges may be either a sequence of 2 numbers, or else a callable that returns such a sequence. ranges[0] corresponds to integration over x0, and so on. If an element of ranges is a callable, then it will be called with all of the integration arguments available, as well as any parametric arguments. e.g., if func = f(x0, x1, x2, t0, t1), then ranges[0] may be defined as either (a, b) or else as (a, b) = range0(x1, x2, t0, t1).

Additional arguments t0, ... tn, required by func, ranges, and opts.

Options to be passed to quad. May be empty, a dict, or a sequence of dicts or functions that return a dict. If empty, the default options from scipy.integrate.quad are used. If a dict, the same options are used for all levels of integraion. If a sequence, then each element of the sequence corresponds to a particular integration. e.g., opts[0] corresponds to integration over x0, and so on. If a callable, the signature must be the same as for ranges. The available options together with their default values are:

For more information on these options, see quad.

Partial implementation of full_output from scipy.integrate.quad. The number of integrand function evaluations neval can be obtained by setting full_output=True when calling nquad.

The result of the integration.

The maximum of the estimates of the absolute error in the various integration results.

A dict containing additional information on the integration.

1-D numerical integration

double and triple integrals

fixed-order Gaussian quadrature

For valid results, the integral must converge; behavior for divergent integrals is not guaranteed.

Details of QUADPACK level routines

nquad calls routines from the FORTRAN library QUADPACK. This section provides details on the conditions for each routine to be called and a short description of each routine. The routine called depends on weight, points and the integration limits a and b.

The following provides a short description from [1] for each routine.

is an integrator based on globally adaptive interval subdivision in connection with extrapolation, which will eliminate the effects of integrand singularities of several types. The integration is is performed using a 21-point Gauss-Kronrod quadrature within each subinterval.

handles integration over infinite intervals. The infinite range is mapped onto a finite interval and subsequently the same strategy as in QAGS is applied.

serves the same purposes as QAGS, but also allows the user to provide explicit information about the location and type of trouble-spots i.e. the abscissae of internal singularities, discontinuities and other difficulties of the integrand function.

is an integrator for the evaluation of \(\int^b_a \cos(\omega x)f(x)dx\) or \(\int^b_a \sin(\omega x)f(x)dx\) over a finite interval [a,b], where \(\omega\) and \(f\) are specified by the user. The rule evaluation component is based on the modified Clenshaw-Curtis technique

An adaptive subdivision scheme is used in connection with an extrapolation procedure, which is a modification of that in QAGS and allows the algorithm to deal with singularities in \(f(x)\).

calculates the Fourier transform \(\int^\infty_a \cos(\omega x)f(x)dx\) or \(\int^\infty_a \sin(\omega x)f(x)dx\) for user-provided \(\omega\) and \(f\). The procedure of QAWO is applied on successive finite intervals, and convergence acceleration by means of the \(\varepsilon\)-algorithm is applied to the series of integral approximations.

approximate \(\int^b_a w(x)f(x)dx\), with \(a < b\) where \(w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)\) with \(\alpha,\beta > -1\), where \(v(x)\) may be one of the following functions: \(1\), \(\log(x-a)\), \(\log(b-x)\), \(\log(x-a)\log(b-x)\).

The user specifies \(\alpha\), \(\beta\) and the type of the function \(v\). A globally adaptive subdivision strategy is applied, with modified Clenshaw-Curtis integration on those subintervals which contain a or b.

compute \(\int^b_a f(x) / (x-c)dx\) where the integral must be interpreted as a Cauchy principal value integral, for user specified \(c\) and \(f\). The strategy is globally adaptive. Modified Clenshaw-Curtis integration is used on those intervals containing the point \(x = c\).

Piessens, Robert; de Doncker-Kapenga, Elise; Überhuber, Christoph W.; Kahaner, David (1983). QUADPACK: A subroutine package for automatic integration. Springer-Verlag. ISBN 978-3-540-12553-2.

and \((t_0, t_1) = (0, 1)\) .

**Examples:**

Example 1 (unknown):
```unknown
double func(int n, double *xx)
double func(int n, double *xx, void *user_data)
```

Example 2 (python):
```python
>>> import numpy as np
>>> from scipy import integrate
>>> func = lambda x0,x1,x2,x3 : x0**2 + x1*x2 - x3**3 + np.sin(x0) + (
...                                 1 if (x0-.2*x3-.5-.25*x1>0) else 0)
>>> def opts0(*args, **kwargs):
...     return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}
>>> integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],
...                 opts=[opts0,{},{},{}], full_output=True)
(1.5267454070738633, 2.9437360001402324e-14, {'neval': 388962})
```

Example 3 (python):
```python
>>> def func2(x0, x1, x2, t0, t1):
...     return x0*x2**2 + np.sin(x1) + 1 + (1 if x0+t1*x1-t0>0 else 0)
>>> def lim0(x1, x2, t0, t1):
...     return [t0*x1 + t1*x2 - 1, t0*x1 + t1*x2 + 1]
>>> def lim1(x2, t0, t1):
...     return [x2 + t0**2*t1**3 - 1, x2 + t0**2*t1**3 + 1]
>>> def lim2(t0, t1):
...     return [t0 + t1 - 1, t0 + t1 + 1]
>>> def opts0(x1, x2, t0, t1):
...     return {'points' : [t0 - t1*x1]}
>>> def opts1(x2, t0, t1):
...     return {}
>>> def opts2(t0, t1):
...     return {}
>>> integrate.nquad(func2, [lim0, lim1, lim2], args=(0,1),
...                 opts=[opts0, opts1, opts2])
(36.099919226771625, 1.8546948553373528e-07)
```

---

## spline_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filter.html

**Contents:**
- spline_filter#

Multidimensional spline filter.

The order of the spline, default is 3.

The array in which to place the output, or the dtype of the returned array. Default is numpy.float64.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘mirror’. Behavior for each valid value is as follows (see additional plots and details on boundary modes):

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

This is a synonym for ‘reflect’.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. No interpolation is performed beyond the edges of the input.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. Interpolation occurs for samples outside the input’s extent as well.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

The input is extended by wrapping around to the opposite edge, but in a way such that the last point and initial point exactly overlap. In this case it is not well defined which sample will be chosen at the point of overlap.

Filtered array. Has the same shape as input.

Calculate a 1-D spline filter along the given axis.

The multidimensional filter is implemented as a sequence of 1-D spline filters. The intermediate arrays are stored in the same data type as the output. Therefore, for output types with a limited precision, the results may be imprecise because intermediate results may be stored with insufficient precision.

For complex-valued input, this function processes the real and imaginary components independently.

Added in version 1.6.0: Complex-valued support added.

We can filter an image using multidimensional splines:

**Examples:**

Example 1 (python):
```python
>>> from scipy.ndimage import spline_filter
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> orig_img = np.eye(20)  # create an image
>>> orig_img[10, :] = 1.0
>>> sp_filter = spline_filter(orig_img, order=3)
>>> f, ax = plt.subplots(1, 2, sharex=True)
>>> for ind, data in enumerate([[orig_img, "original image"],
...                             [sp_filter, "spline filter"]]):
...     ax[ind].imshow(data[0], cmap='gray_r')
...     ax[ind].set_title(data[1])
>>> plt.tight_layout()
>>> plt.show()
```

---

## cho_factor#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_factor.html

**Contents:**
- cho_factor#

Compute the Cholesky decomposition of a matrix, to use in cho_solve

Returns a matrix containing the Cholesky decomposition, A = L L* or A = U* U of a Hermitian positive-definite matrix a. The return value can be directly used as the first parameter to cho_solve.

The returned matrix also contains random data in the entries not used by the Cholesky decomposition. If you need to zero these entries, use the function cholesky instead.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Matrix to be decomposed

Whether to compute the upper or lower triangular Cholesky factorization. During decomposition, only the selected half of the matrix is referenced. (Default: upper-triangular)

Whether to overwrite data in a (may improve performance)

Whether to check that the entire input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Matrix whose upper or lower triangle contains the Cholesky factor of a. Other parts of the matrix contain random data.

Flag indicating whether the factor is in the lower or upper triangle

Raised if decomposition fails.

Solve a linear set equations using the Cholesky factorization of a matrix.

During the finiteness check (if selected), the entire matrix a is checked. During decomposition, a is assumed to be symmetric or Hermitian (as applicable), and only the half selected by option lower is referenced. Consequently, if a is asymmetric/non-Hermitian, cholesky may still succeed if the symmetric/Hermitian matrix represented by the selected half is positive definite, yet it may fail if an element in the other half is non-finite.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import cho_factor
>>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
>>> c, low = cho_factor(A)
>>> c
array([[3.        , 1.        , 0.33333333, 1.66666667],
       [3.        , 2.44948974, 1.90515869, -0.27216553],
       [1.        , 5.        , 2.29330749, 0.8559528 ],
       [5.        , 1.        , 2.        , 1.55418563]])
>>> np.allclose(np.triu(c).T @ np. triu(c) - A, np.zeros((4, 4)))
True
```

---

## Interpolation (scipy.interpolate)#

**URL:** https://docs.scipy.org/doc/scipy/reference/interpolate.html

**Contents:**
- Interpolation (scipy.interpolate)#
- Univariate interpolation#
- Multivariate interpolation#
- 1-D spline smoothing and approximation#
- Rational Approximation#
- Interfaces to FITPACK routines for 1D and 2D spline fitting#
  - 1D FITPACK splines#
  - 2D FITPACK splines#
- Additional tools#

Sub-package for functions and objects used in interpolation.

See the user guide for recommendations on choosing a routine, and other usage details.

make_interp_spline(x, y[, k, t, bc_type, ...])

Create an interpolating B-spline with specified degree and boundary conditions.

CubicSpline(x, y[, axis, bc_type, extrapolate])

Piecewise cubic interpolator to fit values (C2 smooth).

PchipInterpolator(x, y[, axis, extrapolate])

PCHIP shape-preserving interpolator (C1 smooth).

Akima1DInterpolator(x, y[, axis, method, ...])

Akima "visually pleasing" interpolator (C1 smooth).

FloaterHormannInterpolator(points, values, *)

Floater-Hormann barycentric rational interpolator (C∞ smooth on real axis).

BarycentricInterpolator(xi[, yi, axis, wi, ...])

Barycentric (Lagrange with improved stability) interpolator (C∞ smooth).

KroghInterpolator(xi, yi[, axis])

Krogh interpolator (C∞ smooth).

CubicHermiteSpline(x, y, dydx[, axis, ...])

Piecewise cubic interpolator to fit values and first derivatives (C1 smooth).

Low-level data structures for univariate interpolation:

PPoly(c, x[, extrapolate, axis])

Piecewise polynomial in the power basis.

BPoly(c, x[, extrapolate, axis])

Piecewise polynomial in the Bernstein basis.

BSpline(t, c, k[, extrapolate, axis])

Univariate spline in the B-spline basis.

LinearNDInterpolator(points, values[, ...])

Piecewise linear interpolator in N > 1 dimensions.

NearestNDInterpolator(x, y[, rescale, ...])

Nearest-neighbor interpolator in N > 1 dimensions.

CloughTocher2DInterpolator(points, values[, ...])

Piecewise cubic, C1 smooth, curvature-minimizing interpolator in N=2 dimensions.

RBFInterpolator(y, d[, neighbors, ...])

Radial basis function interpolator in N ≥ 1 dimensions.

RegularGridInterpolator(points, values[, ...])

Interpolator of specified order on a rectilinear grid in N ≥ 1 dimensions.

scipy.ndimage.map_coordinates, An example wrapper for map_coordinates

Low-level data structures for tensor product polynomials and splines:

NdPPoly(c, x[, extrapolate])

Piecewise tensor product polynomial

NdBSpline(t, c, k, *[, extrapolate])

Tensor product spline object.

make_lsq_spline(x, y, t[, k, w, axis, ...])

Create a smoothing B-spline satisfying the Least SQuares (LSQ) criterion.

make_smoothing_spline(x, y[, w, lam, axis])

Create a smoothing B-spline satisfying the Generalized Cross Validation (GCV) criterion.

make_splrep(x, y, *[, w, xb, xe, k, s, t, nest])

Create a smoothing B-spline function with bounded error, minimizing derivative jumps.

make_splprep(x, *[, w, u, ub, ue, k, s, t, nest])

Create a smoothing parametric B-spline curve with bounded error, minimizing derivative jumps.

generate_knots(x, y, *[, w, xb, xe, k, s, nest])

Generate knot vectors until the Least SQuares (LSQ) criterion is satified.

AAA(x, y, *[, rtol, max_terms, clean_up, ...])

AAA real or complex rational approximation.

This section lists wrappers for FITPACK functionality for 1D and 2D smoothing splines. In most cases, users are better off using higher-level routines listed in previous sections.

This package provides two sets of functionally equivalent wrappers: object-oriented and functional.

Functional FITPACK interface:

splrep(x, y[, w, xb, xe, k, task, s, t, ...])

Find the B-spline representation of a 1-D curve.

splprep(x[, w, u, ub, ue, k, task, s, t, ...])

Find the B-spline representation of an N-D curve.

splev(x, tck[, der, ext])

Evaluate a B-spline or its derivatives.

splint(a, b, tck[, full_output])

Evaluate the definite integral of a B-spline between two given points.

Find the roots of a cubic B-spline.

Evaluate a B-spline and all its derivatives at one point (or set of points) up to order k (the degree of the spline), being 0 the spline itself.

Compute the spline representation of the derivative of a given spline

Compute the spline for the antiderivative (integral) of a given spline.

insert(x, tck[, m, per])

Insert knots into a B-spline.

Object-oriented FITPACK interface:

UnivariateSpline(x, y[, w, bbox, k, s, ext, ...])

1-D smoothing spline fit to a given set of data points.

InterpolatedUnivariateSpline(x, y[, w, ...])

1-D interpolating spline for a given set of data points.

LSQUnivariateSpline(x, y, t[, w, bbox, k, ...])

1-D spline with explicit internal knots.

RectBivariateSpline(x, y, z[, bbox, kx, ky, ...])

Bivariate spline approximation over a rectangular mesh.

RectSphereBivariateSpline(u, v, r[, s, ...])

Bivariate spline approximation over a rectangular mesh on a sphere.

For unstructured data (OOP interface):

Base class for bivariate splines.

SmoothBivariateSpline(x, y, z[, w, bbox, ...])

Smooth bivariate spline approximation.

SmoothSphereBivariateSpline(theta, phi, r[, ...])

Smooth bivariate spline approximation in spherical coordinates.

LSQBivariateSpline(x, y, z, tx, ty[, w, ...])

Weighted least-squares bivariate spline approximation.

LSQSphereBivariateSpline(theta, phi, r, tt, tp)

Weighted least-squares bivariate spline approximation in spherical coordinates.

For unstructured data (functional interface):

bisplrep(x, y, z[, w, xb, xe, yb, ye, kx, ...])

Find a bivariate B-spline representation of a surface.

bisplev(x, y, tck[, dx, dy])

Evaluate a bivariate B-spline and its derivatives.

Return a Lagrange interpolating polynomial.

approximate_taylor_polynomial(f, x, degree, ...)

Estimate the Taylor polynomial of f at x by polynomial fitting.

Return Pade approximation to a polynomial as the ratio of two polynomials.

interpn(points, values, xi[, method, ...])

Multidimensional interpolation on regular or rectilinear grids.

griddata(points, values, xi[, method, ...])

Convenience function for interpolating unstructured data in multiple dimensions.

barycentric_interpolate(xi, yi, x[, axis, ...])

Convenience function for barycentric interpolation.

krogh_interpolate(xi, yi, x[, der, axis])

Convenience function for Krogh interpolation.

pchip_interpolate(xi, yi, x[, der, axis])

Convenience function for pchip interpolation.

Class for radial basis function interpolation of functions from N-D scattered data to an M-D domain (legacy).

interp1d(x, y[, kind, axis, copy, ...])

Interpolate a 1-D function (legacy).

interp2d(x, y, z[, kind, copy, ...])

Class for 2D interpolation (deprecated and removed)

scipy.ndimage.map_coordinates, scipy.ndimage.spline_filter,

---

## logm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html

**Contents:**
- logm#

Compute matrix logarithm.

The matrix logarithm is the inverse of expm: expm(logm(A)) == A

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Matrix whose logarithm to evaluate

Emit warning if error in the result is estimated large instead of returning estimated error. (Default: True)

Deprecated since version 1.16.0: The disp argument is deprecated and will be removed in SciPy 1.18.0. The previously returned error estimate can be computed as norm(expm(logm(A)) - A, 1) / norm(A, 1).

Matrix logarithm of A

1-norm of the estimated error, ||err||_1 / ||A||_1

Awad H. Al-Mohy and Nicholas J. Higham (2012) “Improved Inverse Scaling and Squaring Algorithms for the Matrix Logarithm.” SIAM Journal on Scientific Computing, 34 (4). C152-C169. ISSN 1095-7197

Nicholas J. Higham (2008) “Functions of Matrices: Theory and Computation” ISBN 978-0-898716-46-7

Nicholas J. Higham and Lijing lin (2011) “A Schur-Pade Algorithm for Fractional Powers of a Matrix.” SIAM Journal on Matrix Analysis and Applications, 32 (3). pp. 1056-1078. ISSN 0895-4798

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import logm, expm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> b = logm(a)
>>> b
array([[-1.02571087,  2.05142174],
       [ 0.68380725,  1.02571087]])
>>> expm(b)         # Verify expm(logm(a)) returns a
array([[ 1.,  3.],
       [ 1.,  4.]])
```

---

## companion#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.companion.html

**Contents:**
- companion#

Create a companion matrix.

Create the companion matrix [1] associated with the polynomial whose coefficients are given in a.

1-D array of polynomial coefficients. The length of a must be at least two, and a[0] must not be zero. M-dimensional arrays are treated as a batch: each slice along the last axis is a 1-D array of polynomial coefficients.

For 1-D input, the first row of c is -a[1:]/a[0], and the first sub-diagonal is all ones. The data-type of the array is the same as the data-type of 1.0*a[0]. For batch input, each slice of shape (N-1, N-1) along the last two dimensions of the output corresponds with a slice of shape (N,) along the last dimension of the input.

If any of the following are true: a) a.shape[-1] < 2; b) a[..., 0] == 0.

Added in version 0.8.0.

R. A. Horn & C. R. Johnson, Matrix Analysis. Cambridge, UK: Cambridge University Press, 1999, pp. 146-7.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import companion
>>> companion([1, -10, 31, -30])
array([[ 10., -31.,  30.],
       [  1.,   0.,   0.],
       [  0.,   1.,   0.]])
```

---

## fftshift#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftshift.html

**Contents:**
- fftshift#

Shift the zero-frequency component to the center of the spectrum.

This function swaps half-spaces for all axes listed (defaults to all). Note that y[0] is the Nyquist component only if len(x) is even.

Axes over which to shift. Default is None, which shifts all axes.

The inverse of fftshift.

Shift the zero-frequency component only along the second axis:

**Examples:**

Example 1 (typescript):
```typescript
>>> import numpy as np
>>> freqs = np.fft.fftfreq(10, 0.1)
>>> freqs
array([ 0.,  1.,  2., ..., -3., -2., -1.])
>>> np.fft.fftshift(freqs)
array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
```

Example 2 (json):
```json
>>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
>>> freqs
array([[ 0.,  1.,  2.],
       [ 3.,  4., -4.],
       [-3., -2., -1.]])
>>> np.fft.fftshift(freqs, axes=(1,))
array([[ 2.,  0.,  1.],
       [-4.,  3.,  4.],
       [-1., -3., -2.]])
```

---

## scipy.special.expn#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expn.html

**Contents:**
- scipy.special.expn#

Generalized exponential integral En.

For integer \(n \geq 0\) and real \(x \geq 0\) the generalized exponential integral is defined as [dlmf]

Non-negative integers

Optional output array for the function results

Values of the generalized exponential integral

special case of \(E_n\) for \(n = 1\)

related to \(E_n\) when \(n = 1\)

Digital Library of Mathematical Functions, 8.19.2 https://dlmf.nist.gov/8.19#E2

Its domain is nonnegative n and x.

It has a pole at x = 0 for n = 1, 2; for larger n it is equal to 1 / (n - 1).

For n equal to 0 it reduces to exp(-x) / x.

For n equal to 1 it reduces to exp1.

**Examples:**

Example 1 (typescript):
```typescript
>>> import numpy as np
>>> import scipy.special as sc
```

Example 2 (unknown):
```unknown
>>> sc.expn(-1, 1.0), sc.expn(1, -1.0)
(nan, nan)
```

Example 3 (unknown):
```unknown
>>> sc.expn([0, 1, 2, 3, 4], 0)
array([       inf,        inf, 1.        , 0.5       , 0.33333333])
```

Example 4 (unknown):
```unknown
>>> x = np.array([1, 2, 3, 4])
>>> sc.expn(0, x)
array([0.36787944, 0.06766764, 0.01659569, 0.00457891])
>>> np.exp(-x) / x
array([0.36787944, 0.06766764, 0.01659569, 0.00457891])
```

---

## norm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html

**Contents:**
- norm#

Matrix or vector norm.

This function is able to return one of eight different matrix norms, or one of an infinite number of vector norms (described below), depending on the value of the ord parameter. For tensors with rank different from 1 or 2, only ord=None is supported.

Input array. If axis is None, a must be 1-D or 2-D, unless ord is None. If both axis and ord are None, the 2-norm of a.ravel will be returned.

Order of the norm (see table under Notes). inf means NumPy’s inf object.

If axis is an integer, it specifies the axis of a along which to compute the vector norms. If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed. If axis is None then either a vector norm (when a is 1-D) or a matrix norm (when a is 2-D) is returned.

If this is set to True, the axes which are normed over are left in the result as dimensions with size one. With this option the result will broadcast correctly against the original a.

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Norm of the matrix or vector(s).

For values of ord <= 0, the result is, strictly speaking, not a mathematical ‘norm’, but it may still be useful for various numerical purposes.

The following norms can be calculated:

max(sum(abs(a), axis=1))

min(sum(abs(a), axis=1))

max(sum(abs(a), axis=0))

min(sum(abs(a), axis=0))

2-norm (largest sing. value)

smallest singular value

sum(abs(a)**ord)**(1./ord)

The Frobenius norm is given by [1]:

\(||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}\)

The nuclear norm is the sum of the singular values.

Both the Frobenius and nuclear norm orders are only defined for matrices.

G. H. Golub and C. F. Van Loan, Matrix Computations, Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import norm
>>> a = np.arange(9) - 4.0
>>> a
array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
>>> b = a.reshape((3, 3))
>>> b
array([[-4., -3., -2.],
       [-1.,  0.,  1.],
       [ 2.,  3.,  4.]])
```

Example 2 (unknown):
```unknown
>>> norm(a)
7.745966692414834
>>> norm(b)
7.745966692414834
>>> norm(b, 'fro')
7.745966692414834
>>> norm(a, np.inf)
4.0
>>> norm(b, np.inf)
9.0
>>> norm(a, -np.inf)
0.0
>>> norm(b, -np.inf)
2.0
```

Example 3 (unknown):
```unknown
>>> norm(a, 1)
20.0
>>> norm(b, 1)
7.0
>>> norm(a, -1)
-4.6566128774142013e-010
>>> norm(b, -1)
6.0
>>> norm(a, 2)
7.745966692414834
>>> norm(b, 2)
7.3484692283495345
```

Example 4 (unknown):
```unknown
>>> norm(a, -2)
0.0
>>> norm(b, -2)
1.8570331885190563e-016
>>> norm(a, 3)
5.8480354764257312
>>> norm(a, -3)
0.0
```

---

## circulant#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.circulant.html

**Contents:**
- circulant#

Construct a circulant matrix.

The first column(s) of the matrix. Multidimensional arrays are treated as a batch: each slice along the last axis is the first column of an output matrix.

A circulant matrix whose first column is given by c. For batch input, each slice of shape (N, N) along the last two dimensions of the output corresponds with a slice of shape (N,) along the last dimension of the input.

Solve a circulant system.

Added in version 0.8.0.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import circulant
>>> circulant([1, 2, 3])
array([[1, 3, 2],
       [2, 1, 3],
       [3, 2, 1]])
```

Example 2 (json):
```json
>>> circulant([[1, 2, 3], [4, 5, 6]])
array([[[1, 3, 2],
        [2, 1, 3],
        [3, 2, 1]],
       [[4, 6, 5],
        [5, 4, 6],
        [6, 5, 4]]])
```

---

## Contingency table functions (scipy.stats.contingency)#

**URL:** https://docs.scipy.org/doc/scipy/reference/stats.contingency.html

**Contents:**
- Contingency table functions (scipy.stats.contingency)#

Functions for creating and analyzing contingency tables.

chi2_contingency(observed[, correction, ...])

Chi-square test of independence of variables in a contingency table.

relative_risk(exposed_cases, exposed_total, ...)

Compute the relative risk (also known as the risk ratio).

odds_ratio(table, *[, kind])

Compute the odds ratio for a 2x2 contingency table.

crosstab(*args[, levels, sparse])

Return table of counts for each possible unique combination in *args.

association(observed[, method, correction, ...])

Calculates degree of association between two nominal variables.

expected_freq(observed)

Compute the expected frequencies from a contingency table.

Return a list of the marginal sums of the array a.

---

## dblquad#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html

**Contents:**
- dblquad#

Compute a double integral.

Return the double (definite) integral of func(y, x) from x = a..b and y = gfun(x)..hfun(x).

A Python function or method of at least two variables: y must be the first argument and x the second argument.

The limits of integration in x: a < b

The lower boundary curve in y which is a function taking a single floating point argument (x) and returning a floating point result or a float indicating a constant boundary curve.

The upper boundary curve in y (same requirements as gfun).

Extra arguments to pass to func.

Absolute tolerance passed directly to the inner 1-D quadrature integration. Default is 1.49e-8. dblquad tries to obtain an accuracy of abs(i-result) <= max(epsabs, epsrel*abs(i)) where i = inner integral of func(y, x) from gfun(x) to hfun(x), and result is the numerical approximation. See epsrel below.

Relative tolerance of the inner 1-D integrals. Default is 1.49e-8. If epsabs <= 0, epsrel must be greater than both 5e-29 and 50 * (machine epsilon). See epsabs above.

The resultant integral.

An estimate of the error.

N-dimensional integrals

fixed-order Gaussian quadrature

integrator for sampled data

integrator for sampled data

for coefficients and roots of orthogonal polynomials

For valid results, the integral must converge; behavior for divergent integrals is not guaranteed.

Details of QUADPACK level routines

quad calls routines from the FORTRAN library QUADPACK. This section provides details on the conditions for each routine to be called and a short description of each routine. For each level of integration, qagse is used for finite limits or qagie is used if either limit (or both!) are infinite. The following provides a short description from [1] for each routine.

is an integrator based on globally adaptive interval subdivision in connection with extrapolation, which will eliminate the effects of integrand singularities of several types. The integration is is performed using a 21-point Gauss-Kronrod quadrature within each subinterval.

handles integration over infinite intervals. The infinite range is mapped onto a finite interval and subsequently the same strategy as in QAGS is applied.

Piessens, Robert; de Doncker-Kapenga, Elise; Überhuber, Christoph W.; Kahaner, David (1983). QUADPACK: A subroutine package for automatic integration. Springer-Verlag. ISBN 978-3-540-12553-2.

Compute the double integral of x * y**2 over the box x ranging from 0 to 2 and y ranging from 0 to 1. That is, \(\int^{x=2}_{x=0} \int^{y=1}_{y=0} x y^2 \,dy \,dx\).

Calculate \(\int^{x=\pi/4}_{x=0} \int^{y=\cos(x)}_{y=\sin(x)} 1 \,dy \,dx\).

Calculate \(\int^{x=1}_{x=0} \int^{y=2-x}_{y=x} a x y \,dy \,dx\) for \(a=1, 3\).

Compute the two-dimensional Gaussian Integral, which is the integral of the Gaussian function \(f(x,y) = e^{-(x^{2} + y^{2})}\), over \((-\infty,+\infty)\). That is, compute the integral \(\iint^{+\infty}_{-\infty} e^{-(x^{2} + y^{2})} \,dy\,dx\).

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import integrate
>>> f = lambda y, x: x*y**2
>>> integrate.dblquad(f, 0, 2, 0, 1)
    (0.6666666666666667, 7.401486830834377e-15)
```

Example 2 (json):
```json
>>> f = lambda y, x: 1
>>> integrate.dblquad(f, 0, np.pi/4, np.sin, np.cos)
    (0.41421356237309503, 1.1083280054755938e-14)
```

Example 3 (json):
```json
>>> f = lambda y, x, a: a*x*y
>>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(1,))
    (0.33333333333333337, 5.551115123125783e-15)
>>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(3,))
    (0.9999999999999999, 1.6653345369377348e-14)
```

Example 4 (unknown):
```unknown
>>> f = lambda x, y: np.exp(-(x ** 2 + y ** 2))
>>> integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
    (3.141592653589777, 2.5173086737433208e-08)
```

---

## geometric_transform#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.geometric_transform.html

**Contents:**
- geometric_transform#

Apply an arbitrary geometric transform.

The given mapping function is used to find, for each point in the output, the corresponding coordinates in the input. The value of the input at those coordinates is determined by spline interpolation of the requested order.

A callable object that accepts a tuple of length equal to the output array rank, and returns the corresponding input coordinates as a tuple of length equal to the input array rank.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘constant’. Behavior for each valid value is as follows (see additional plots and details on boundary modes):

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

This is a synonym for ‘reflect’.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. No interpolation is performed beyond the edges of the input.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. Interpolation occurs for samples outside the input’s extent as well.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

The input is extended by wrapping around to the opposite edge, but in a way such that the last point and initial point exactly overlap. In this case it is not well defined which sample will be chosen at the point of overlap.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Determines if the input array is prefiltered with spline_filter before interpolation. The default is True, which will create a temporary float64 array of filtered values if order > 1. If setting this to False, the output will be slightly blurred if order > 1, unless the input is prefiltered, i.e. it is the result of calling spline_filter on the original input.

Extra arguments passed to mapping.

Extra keywords passed to mapping.

This function also accepts low-level callback functions with one the following signatures and wrapped in scipy.LowLevelCallable:

The calling function iterates over the elements of the output array, calling the callback function at each element. The coordinates of the current output element are passed through output_coordinates. The callback function must return the coordinates at which the input must be interpolated in input_coordinates. The rank of the input and output arrays are given by input_rank and output_rank respectively. user_data is the data pointer provided to scipy.LowLevelCallable as-is.

The callback function must return an integer error status that is zero if something went wrong and one otherwise. If an error occurs, you should normally set the Python error status with an informative message before returning, otherwise a default error message is set by the calling function.

In addition, some other low-level function pointer specifications are accepted, but these are for backward compatibility only and should not be used in new code.

For complex-valued input, this function transforms the real and imaginary components independently.

Added in version 1.6.0: Complex-valued support added.

**Examples:**

Example 1 (unknown):
```unknown
int mapping(npy_intp *output_coordinates, double *input_coordinates,
            int output_rank, int input_rank, void *user_data)
int mapping(intptr_t *output_coordinates, double *input_coordinates,
            int output_rank, int input_rank, void *user_data)
```

Example 2 (python):
```python
>>> import numpy as np
>>> from scipy.ndimage import geometric_transform
>>> a = np.arange(12.).reshape((4, 3))
>>> def shift_func(output_coords):
...     return (output_coords[0] - 0.5, output_coords[1] - 0.5)
...
>>> geometric_transform(a, shift_func)
array([[ 0.   ,  0.   ,  0.   ],
       [ 0.   ,  1.362,  2.738],
       [ 0.   ,  4.812,  6.187],
       [ 0.   ,  8.263,  9.637]])
```

Example 3 (python):
```python
>>> b = [1, 2, 3, 4, 5]
>>> def shift_func(output_coords):
...     return (output_coords[0] - 3,)
...
>>> geometric_transform(b, shift_func, mode='constant')
array([0, 0, 0, 1, 2])
>>> geometric_transform(b, shift_func, mode='nearest')
array([1, 1, 1, 1, 2])
>>> geometric_transform(b, shift_func, mode='reflect')
array([3, 2, 1, 1, 2])
>>> geometric_transform(b, shift_func, mode='wrap')
array([2, 3, 4, 1, 2])
```

---

## lu_solve#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html

**Contents:**
- lu_solve#

Solve an equation system, a x = b, given the LU factorization of a

Factorization of the coefficient matrix a, as given by lu_factor. In particular piv are 0-indexed pivot indices.

Type of system to solve:

Whether to overwrite data in b (may increase performance)

Whether to check that the input matrices contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Solution to the system

LU factorize a matrix

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import lu_factor, lu_solve
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> b = np.array([1, 1, 1, 1])
>>> lu, piv = lu_factor(A)
>>> x = lu_solve((lu, piv), b)
>>> np.allclose(A @ x - b, np.zeros((4,)))
True
```

---

## lu#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html

**Contents:**
- lu#

Compute LU decomposition of a matrix with partial pivoting.

The decomposition satisfies:

where P is a permutation matrix, L lower triangular with unit diagonal elements, and U upper triangular. If permute_l is set to True then L is returned already permuted and hence satisfying A = L @ U.

Perform the multiplication P*L (Default: do not permute)

Whether to overwrite data in a (may improve performance)

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

If True the permutation information is returned as row indices. The default is False for backwards-compatibility reasons.

Permutation arrays or vectors depending on p_indices

Lower triangular or trapezoidal array with unit diagonal. K = min(M, N)

Upper triangular or trapezoidal array

Permuted L matrix. K = min(M, N)

Upper triangular or trapezoidal array

Permutation matrices are costly since they are nothing but row reorder of L and hence indices are strongly recommended to be used instead if the permutation is required. The relation in the 2D case then becomes simply A = L[P, :] @ U. In higher dimensions, it is better to use permute_l to avoid complicated indexing tricks.

In 2D case, if one has the indices however, for some reason, the permutation matrix is still needed then it can be constructed by np.eye(M)[P, :].

We can also use nd-arrays, for example, a demonstration with 4D array:

**Examples:**

Example 1 (unknown):
```unknown
A = P @ L @ U
```

Example 2 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import lu
>>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
>>> p, l, u = lu(A)
>>> np.allclose(A, p @ l @ u)
True
>>> p  # Permutation matrix
array([[0., 1., 0., 0.],  # Row index 1
       [0., 0., 0., 1.],  # Row index 3
       [1., 0., 0., 0.],  # Row index 0
       [0., 0., 1., 0.]]) # Row index 2
>>> p, _, _ = lu(A, p_indices=True)
>>> p
array([1, 3, 0, 2], dtype=int32)  # as given by row indices above
>>> np.allclose(A, l[p, :] @ u)
True
```

Example 3 (unknown):
```unknown
>>> rng = np.random.default_rng()
>>> A = rng.uniform(low=-4, high=4, size=[3, 2, 4, 8])
>>> p, l, u = lu(A)
>>> p.shape, l.shape, u.shape
((3, 2, 4, 4), (3, 2, 4, 4), (3, 2, 4, 8))
>>> np.allclose(A, p @ l @ u)
True
>>> PL, U = lu(A, permute_l=True)
>>> np.allclose(A, PL @ U)
True
```

---

## odeint#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

**Contents:**
- odeint#

Integrate a system of ordinary differential equations.

For new code, use scipy.integrate.solve_ivp to solve a differential equation.

Solve a system of ordinary differential equations using lsoda from the FORTRAN library odepack.

Solves the initial value problem for stiff or non-stiff systems of first order ode-s:

where y can be a vector.

By default, the required order of the first two arguments of func are in the opposite order of the arguments in the system definition function used by the scipy.integrate.ode class and the function scipy.integrate.solve_ivp. To use a function with the signature func(t, y, ...), the argument tfirst must be set to True.

Computes the derivative of y at t. If the signature is callable(t, y, ...), then the argument tfirst must be set True. func must not modify the data in y, as it is a view of the data used internally by the ODE solver.

Initial condition on y (can be a vector).

A sequence of time points for which to solve for y. The initial value point should be the first element of this sequence. This sequence must be monotonically increasing or monotonically decreasing; repeated values are allowed.

Extra arguments to pass to function.

Gradient (Jacobian) of func. If the signature is callable(t, y, ...), then the argument tfirst must be set True. Dfun must not modify the data in y, as it is a view of the data used internally by the ODE solver.

True if Dfun defines derivatives down columns (faster), otherwise Dfun should define derivatives across rows.

True if to return a dictionary of optional outputs as the second output

Whether to print the convergence message

If True, the first two arguments of func (and Dfun, if given) must t, y instead of the default y, t.

Added in version 1.1.0.

Array containing the value of y for each desired time in t, with the initial value y0 in the first row.

Dictionary containing additional output information

vector of step sizes successfully used for each time step

vector with the value of t reached for each time step (will always be at least as large as the input times)

vector of tolerance scale factors, greater than 1.0, computed when a request for too much accuracy was detected

value of t at the time of the last method switch (given for each time step)

cumulative number of time steps

cumulative number of function evaluations for each time step

cumulative number of jacobian evaluations for each time step

a vector of method orders for each successful step

index of the component of largest magnitude in the weighted local error vector (e / ewt) on an error return, -1 otherwise

the length of the double work array required

the length of integer work array required

a vector of method indicators for each successful time step: 1: adams (nonstiff), 2: bdf (stiff)

If either of these are not None or non-negative, then the Jacobian is assumed to be banded. These give the number of lower and upper non-zero diagonals in this banded matrix. For the banded case, Dfun should return a matrix whose rows contain the non-zero bands (starting with the lowest diagonal). Thus, the return matrix jac from Dfun should have shape (ml + mu + 1, len(y0)) when ml >=0 or mu >=0. The data in jac must be stored such that jac[i - j + mu, j] holds the derivative of the ith equation with respect to the jth state variable. If col_deriv is True, the transpose of this jac must be returned.

The input parameters rtol and atol determine the error control performed by the solver. The solver will control the vector, e, of estimated local errors in y, according to an inequality of the form max-norm of (e / ewt) <= 1, where ewt is a vector of positive error weights computed as ewt = rtol * abs(y) + atol. rtol and atol can be either vectors the same length as y or scalars. Defaults to 1.49012e-8.

Vector of critical points (e.g., singularities) where integration care should be taken.

The step size to be attempted on the first step.

The maximum absolute step size allowed.

The minimum absolute step size allowed.

Whether to generate extra printing at method switches.

Maximum number of (internally defined) steps allowed for each integration point in t.

Maximum number of messages printed.

Maximum order to be allowed for the non-stiff (Adams) method.

Maximum order to be allowed for the stiff (BDF) method.

solve an initial value problem for a system of ODEs

a more object-oriented integrator based on VODE

for finding the area under a curve

The second order differential equation for the angle theta of a pendulum acted on by gravity with friction can be written:

where b and c are positive constants, and a prime (’) denotes a derivative. To solve this equation with odeint, we must first convert it to a system of first order equations. By defining the angular velocity omega(t) = theta'(t), we obtain the system:

Let y be the vector [theta, omega]. We implement this system in Python as:

We assume the constants are b = 0.25 and c = 5.0:

For initial conditions, we assume the pendulum is nearly vertical with theta(0) = pi - 0.1, and is initially at rest, so omega(0) = 0. Then the vector of initial conditions is

We will generate a solution at 101 evenly spaced samples in the interval 0 <= t <= 10. So our array of times is:

Call odeint to generate the solution. To pass the parameters b and c to pend, we give them to odeint using the args argument.

The solution is an array with shape (101, 2). The first column is theta(t), and the second is omega(t). The following code plots both components.

**Examples:**

Example 1 (unknown):
```unknown
dy/dt = func(y, t, ...)  [or func(t, y, ...)]
```

Example 2 (unknown):
```unknown
theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0
```

Example 3 (unknown):
```unknown
theta'(t) = omega(t)
omega'(t) = -b*omega(t) - c*sin(theta(t))
```

Example 4 (python):
```python
>>> import numpy as np
>>> def pend(y, t, b, c):
...     theta, omega = y
...     dydt = [omega, -b*omega - c*np.sin(theta)]
...     return dydt
...
```

---

## Integration and ODEs (scipy.integrate)#

**URL:** https://docs.scipy.org/doc/scipy/reference/integrate.html

**Contents:**
- Integration and ODEs (scipy.integrate)#
- Integrating functions, given function object#
- Integrating functions, given fixed samples#
- Summation#
- Solving initial value problems for ODE systems#
  - Old API#
- Solving boundary value problems for ODE systems#

quad(func, a, b[, args, full_output, ...])

Compute a definite integral.

quad_vec(f, a, b[, epsabs, epsrel, norm, ...])

Adaptive integration of a vector-valued function.

cubature(f, a, b, *[, rule, rtol, atol, ...])

Adaptive cubature of multidimensional array-valued function.

dblquad(func, a, b, gfun, hfun[, args, ...])

Compute a double integral.

tplquad(func, a, b, gfun, hfun, qfun, rfun)

Compute a triple (definite) integral.

nquad(func, ranges[, args, opts, full_output])

Integration over multiple variables.

tanhsinh(f, a, b, *[, args, log, maxlevel, ...])

Evaluate a convergent integral numerically using tanh-sinh quadrature.

fixed_quad(func, a, b[, args, n])

Compute a definite integral using fixed-order Gaussian quadrature.

newton_cotes(rn[, equal])

Return weights and error coefficient for Newton-Cotes integration.

qmc_quad(func, a, b, *[, n_estimates, ...])

Compute an integral in N-dimensions using Quasi-Monte Carlo quadrature.

Warning on issues during integration.

trapezoid(y[, x, dx, axis])

Integrate along the given axis using the composite trapezoidal rule.

cumulative_trapezoid(y[, x, dx, axis, initial])

Cumulatively integrate y(x) using the composite trapezoidal rule.

simpson(y[, x, dx, axis])

Integrate y(x) using samples along the given axis and the composite Simpson's rule.

cumulative_simpson(y, *[, x, dx, axis, initial])

Cumulatively integrate y(x) using the composite Simpson's 1/3 rule.

romb(y[, dx, axis, show])

Romberg integration using samples of a function.

scipy.special for orthogonal polynomials (special) for Gaussian quadrature roots and weights for other weighting factors and regions.

nsum(f, a, b, *[, step, args, log, ...])

Evaluate a convergent finite or infinite series.

The solvers are implemented as individual classes, which can be used directly (low-level usage) or through a convenience function.

solve_ivp(fun, t_span, y0[, method, t_eval, ...])

Solve an initial value problem for a system of ODEs.

RK23(fun, t0, y0, t_bound[, max_step, rtol, ...])

Explicit Runge-Kutta method of order 3(2).

RK45(fun, t0, y0, t_bound[, max_step, rtol, ...])

Explicit Runge-Kutta method of order 5(4).

DOP853(fun, t0, y0, t_bound[, max_step, ...])

Explicit Runge-Kutta method of order 8.

Radau(fun, t0, y0, t_bound[, max_step, ...])

Implicit Runge-Kutta method of Radau IIA family of order 5.

BDF(fun, t0, y0, t_bound[, max_step, rtol, ...])

Implicit method based on backward-differentiation formulas.

LSODA(fun, t0, y0, t_bound[, first_step, ...])

Adams/BDF method with automatic stiffness detection and switching.

OdeSolver(fun, t0, y0, t_bound, vectorized)

Base class for ODE solvers.

DenseOutput(t_old, t)

Base class for local interpolant over step made by an ODE solver.

OdeSolution(ts, interpolants[, alt_segment])

Continuous ODE solution.

These are the routines developed earlier for SciPy. They wrap older solvers implemented in Fortran (mostly ODEPACK). While the interface to them is not particularly convenient and certain features are missing compared to the new API, the solvers themselves are of good quality and work fast as compiled Fortran code. In some cases, it might be worth using this old API.

odeint(func, y0, t[, args, Dfun, col_deriv, ...])

Integrate a system of ordinary differential equations.

A generic interface class to numeric integrators.

complex_ode(f[, jac])

A wrapper of ode for complex systems.

Warning raised during the execution of odeint.

solve_bvp(fun, bc, x, y[, p, S, fun_jac, ...])

Solve a boundary value problem for a system of ODEs.

---

## Low-level LAPACK functions (scipy.linalg.lapack)#

**URL:** https://docs.scipy.org/doc/scipy/reference/linalg.lapack.html

**Contents:**
- Low-level LAPACK functions (scipy.linalg.lapack)#
- Finding functions#
- All functions#

This module contains low-level functions from the LAPACK library.

Added in version 0.12.0.

The common overwrite_<> option in many routines, allows the input arrays to be overwritten to avoid extra memory allocation. However this requires the array to satisfy two conditions which are memory order and the data type to match exactly the order and the type expected by the routine.

As an example, if you pass a double precision float array to any S.... routine which expects single precision arguments, f2py will create an intermediate array to match the argument types and overwriting will be performed on that intermediate array.

Similarly, if a C-contiguous array is passed, f2py will pass a FORTRAN-contiguous array internally. Please make sure that these details are satisfied. More information can be found in the f2py documentation.

These functions do little to no error checking. It is possible to cause crashes by mis-using them, so prefer using the higher-level routines in scipy.linalg.

get_lapack_funcs(names[, arrays, dtype, ilp64])

Return available LAPACK function objects from names.

sgbcon(kl,ku,ab,ipiv,anorm,[norm,ldab])

dgbcon(kl,ku,ab,ipiv,anorm,[norm,ldab])

cgbcon(kl,ku,ab,ipiv,anorm,[norm,ldab])

zgbcon(kl,ku,ab,ipiv,anorm,[norm,ldab])

sgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])

dgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])

cgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])

zgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])

sgbtrf(ab,kl,ku,[m,n,ldab,overwrite_ab])

dgbtrf(ab,kl,ku,[m,n,ldab,overwrite_ab])

cgbtrf(ab,kl,ku,[m,n,ldab,overwrite_ab])

zgbtrf(ab,kl,ku,[m,n,ldab,overwrite_ab])

sgebal(a,[scale,permute,overwrite_a])

dgebal(a,[scale,permute,overwrite_a])

cgebal(a,[scale,permute,overwrite_a])

zgebal(a,[scale,permute,overwrite_a])

sgecon(a,anorm,[norm])

dgecon(a,anorm,[norm])

cgecon(a,anorm,[norm])

zgecon(a,anorm,[norm])

sgeev_lwork(n,[compute_vl,compute_vr])

Wrapper for sgeev_lwork.

dgeev_lwork(n,[compute_vl,compute_vr])

Wrapper for dgeev_lwork.

cgeev_lwork(n,[compute_vl,compute_vr])

Wrapper for cgeev_lwork.

zgeev_lwork(n,[compute_vl,compute_vr])

Wrapper for zgeev_lwork.

sgehrd(a,[lo,hi,lwork,overwrite_a])

dgehrd(a,[lo,hi,lwork,overwrite_a])

cgehrd(a,[lo,hi,lwork,overwrite_a])

zgehrd(a,[lo,hi,lwork,overwrite_a])

sgehrd_lwork(n,[lo,hi])

Wrapper for sgehrd_lwork.

dgehrd_lwork(n,[lo,hi])

Wrapper for dgehrd_lwork.

cgehrd_lwork(n,[lo,hi])

Wrapper for cgehrd_lwork.

zgehrd_lwork(n,[lo,hi])

Wrapper for zgehrd_lwork.

sgels(a,b,[trans,lwork,overwrite_a,overwrite_b])

dgels(a,b,[trans,lwork,overwrite_a,overwrite_b])

cgels(a,b,[trans,lwork,overwrite_a,overwrite_b])

zgels(a,b,[trans,lwork,overwrite_a,overwrite_b])

sgels_lwork(m,n,nrhs,[trans])

Wrapper for sgels_lwork.

dgels_lwork(m,n,nrhs,[trans])

Wrapper for dgels_lwork.

cgels_lwork(m,n,nrhs,[trans])

Wrapper for cgels_lwork.

zgels_lwork(m,n,nrhs,[trans])

Wrapper for zgels_lwork.

sgelsd_lwork(m,n,nrhs,[cond,lwork])

Wrapper for sgelsd_lwork.

dgelsd_lwork(m,n,nrhs,[cond,lwork])

Wrapper for dgelsd_lwork.

cgelsd_lwork(m,n,nrhs,[cond,lwork])

Wrapper for cgelsd_lwork.

zgelsd_lwork(m,n,nrhs,[cond,lwork])

Wrapper for zgelsd_lwork.

sgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])

dgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])

cgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])

zgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])

sgelss_lwork(m,n,nrhs,[cond,lwork])

Wrapper for sgelss_lwork.

dgelss_lwork(m,n,nrhs,[cond,lwork])

Wrapper for dgelss_lwork.

cgelss_lwork(m,n,nrhs,[cond,lwork])

Wrapper for cgelss_lwork.

zgelss_lwork(m,n,nrhs,[cond,lwork])

Wrapper for zgelss_lwork.

sgelsy_lwork(m,n,nrhs,cond,[lwork])

Wrapper for sgelsy_lwork.

dgelsy_lwork(m,n,nrhs,cond,[lwork])

Wrapper for dgelsy_lwork.

cgelsy_lwork(m,n,nrhs,cond,[lwork])

Wrapper for cgelsy_lwork.

zgelsy_lwork(m,n,nrhs,cond,[lwork])

Wrapper for zgelsy_lwork.

sgeqp3(a,[lwork,overwrite_a])

dgeqp3(a,[lwork,overwrite_a])

cgeqp3(a,[lwork,overwrite_a])

zgeqp3(a,[lwork,overwrite_a])

sgeqrf(a,[lwork,overwrite_a])

dgeqrf(a,[lwork,overwrite_a])

cgeqrf(a,[lwork,overwrite_a])

zgeqrf(a,[lwork,overwrite_a])

Wrapper for sgeqrf_lwork.

Wrapper for dgeqrf_lwork.

Wrapper for cgeqrf_lwork.

Wrapper for zgeqrf_lwork.

sgeqrfp(a,[lwork,overwrite_a])

dgeqrfp(a,[lwork,overwrite_a])

cgeqrfp(a,[lwork,overwrite_a])

zgeqrfp(a,[lwork,overwrite_a])

Wrapper for sgeqrfp_lwork.

Wrapper for dgeqrfp_lwork.

Wrapper for cgeqrfp_lwork.

Wrapper for zgeqrfp_lwork.

sgerqf(a,[lwork,overwrite_a])

dgerqf(a,[lwork,overwrite_a])

cgerqf(a,[lwork,overwrite_a])

zgerqf(a,[lwork,overwrite_a])

sgesdd_lwork(m,n,[compute_uv,full_matrices])

Wrapper for sgesdd_lwork.

dgesdd_lwork(m,n,[compute_uv,full_matrices])

Wrapper for dgesdd_lwork.

cgesdd_lwork(m,n,[compute_uv,full_matrices])

Wrapper for cgesdd_lwork.

zgesdd_lwork(m,n,[compute_uv,full_matrices])

Wrapper for zgesdd_lwork.

sgesv(a,b,[overwrite_a,overwrite_b])

dgesv(a,b,[overwrite_a,overwrite_b])

cgesv(a,b,[overwrite_a,overwrite_b])

zgesv(a,b,[overwrite_a,overwrite_b])

sgesvd_lwork(m,n,[compute_uv,full_matrices])

Wrapper for sgesvd_lwork.

dgesvd_lwork(m,n,[compute_uv,full_matrices])

Wrapper for dgesvd_lwork.

cgesvd_lwork(m,n,[compute_uv,full_matrices])

Wrapper for cgesvd_lwork.

zgesvd_lwork(m,n,[compute_uv,full_matrices])

Wrapper for zgesvd_lwork.

sgetrf(a,[overwrite_a])

dgetrf(a,[overwrite_a])

cgetrf(a,[overwrite_a])

zgetrf(a,[overwrite_a])

sgetc2(a,[overwrite_a])

dgetc2(a,[overwrite_a])

cgetc2(a,[overwrite_a])

zgetc2(a,[overwrite_a])

sgetri(lu,piv,[lwork,overwrite_lu])

dgetri(lu,piv,[lwork,overwrite_lu])

cgetri(lu,piv,[lwork,overwrite_lu])

zgetri(lu,piv,[lwork,overwrite_lu])

Wrapper for sgetri_lwork.

Wrapper for dgetri_lwork.

Wrapper for cgetri_lwork.

Wrapper for zgetri_lwork.

sgetrs(lu,piv,b,[trans,overwrite_b])

dgetrs(lu,piv,b,[trans,overwrite_b])

cgetrs(lu,piv,b,[trans,overwrite_b])

zgetrs(lu,piv,b,[trans,overwrite_b])

sgesc2(lu,rhs,ipiv,jpiv,[overwrite_rhs])

dgesc2(lu,rhs,ipiv,jpiv,[overwrite_rhs])

cgesc2(lu,rhs,ipiv,jpiv,[overwrite_rhs])

zgesc2(lu,rhs,ipiv,jpiv,[overwrite_rhs])

sgglse_lwork(m, n, p)

Wrapper for sgglse_lwork.

dgglse_lwork(m, n, p)

Wrapper for dgglse_lwork.

cgglse_lwork(m, n, p)

Wrapper for cgglse_lwork.

zgglse_lwork(m, n, p)

Wrapper for zgglse_lwork.

checon(a,ipiv,anorm,[lower])

zhecon(a,ipiv,anorm,[lower])

cheev(a,[compute_v,lower,lwork,overwrite_a])

zheev(a,[compute_v,lower,lwork,overwrite_a])

cheev_lwork(n,[lower])

Wrapper for cheev_lwork.

zheev_lwork(n,[lower])

Wrapper for zheev_lwork.

cheevd_lwork(n,[compute_v,lower])

Wrapper for cheevd_lwork.

zheevd_lwork(n,[compute_v,lower])

Wrapper for zheevd_lwork.

cheevr_lwork(n,[lower])

Wrapper for cheevr_lwork.

zheevr_lwork(n,[lower])

Wrapper for zheevr_lwork.

cheevx_lwork(n,[lower])

Wrapper for cheevx_lwork.

zheevx_lwork(n,[lower])

Wrapper for zheevx_lwork.

chegst(a,b,[itype,lower,overwrite_a])

zhegst(a,b,[itype,lower,overwrite_a])

chegv_lwork(n,[uplo])

Wrapper for chegv_lwork.

zhegv_lwork(n,[uplo])

Wrapper for zhegv_lwork.

chegvx_lwork(n,[uplo])

Wrapper for chegvx_lwork.

zhegvx_lwork(n,[uplo])

Wrapper for zhegvx_lwork.

chesv(a,b,[lwork,lower,overwrite_a,overwrite_b])

zhesv(a,b,[lwork,lower,overwrite_a,overwrite_b])

chesv_lwork(n,[lower])

Wrapper for chesv_lwork.

zhesv_lwork(n,[lower])

Wrapper for zhesv_lwork.

chesvx_lwork(n,[lower])

Wrapper for chesvx_lwork.

zhesvx_lwork(n,[lower])

Wrapper for zhesvx_lwork.

chetrd(a,[lower,lwork,overwrite_a])

zhetrd(a,[lower,lwork,overwrite_a])

chetrd_lwork(n,[lower])

Wrapper for chetrd_lwork.

zhetrd_lwork(n,[lower])

Wrapper for zhetrd_lwork.

chetrf(a,[lower,lwork,overwrite_a])

zhetrf(a,[lower,lwork,overwrite_a])

chetrf_lwork(n,[lower])

Wrapper for chetrf_lwork.

zhetrf_lwork(n,[lower])

Wrapper for zhetrf_lwork.

chetri(a,ipiv,[lower,overwrite_a])

zhetri(a,ipiv,[lower,overwrite_a])

chetrs(a,ipiv,b,[lower,overwrite_b])

zhetrs(a,ipiv,b,[lower,overwrite_b])

slangb(norm,kl,ku,ab,[ldab])

dlangb(norm,kl,ku,ab,[ldab])

clangb(norm,kl,ku,ab,[ldab])

zlangb(norm,kl,ku,ab,[ldab])

slantr(norm,a,[uplo,diag])

dlantr(norm,a,[uplo,diag])

clantr(norm,a,[uplo,diag])

zlantr(norm,a,[uplo,diag])

slarf(v,tau,c,work,[side,incv,overwrite_c])

dlarf(v,tau,c,work,[side,incv,overwrite_c])

clarf(v,tau,c,work,[side,incv,overwrite_c])

zlarf(v,tau,c,work,[side,incv,overwrite_c])

slarfg(n,alpha,x,[incx,overwrite_x])

dlarfg(n,alpha,x,[incx,overwrite_x])

clarfg(n,alpha,x,[incx,overwrite_x])

zlarfg(n,alpha,x,[incx,overwrite_x])

slaswp(a,piv,[k1,k2,off,inc,overwrite_a])

dlaswp(a,piv,[k1,k2,off,inc,overwrite_a])

claswp(a,piv,[k1,k2,off,inc,overwrite_a])

zlaswp(a,piv,[k1,k2,off,inc,overwrite_a])

slauum(c,[lower,overwrite_c])

dlauum(c,[lower,overwrite_c])

clauum(c,[lower,overwrite_c])

zlauum(c,[lower,overwrite_c])

sorcsd_lwork(m, p, q)

Wrapper for sorcsd_lwork.

dorcsd_lwork(m, p, q)

Wrapper for dorcsd_lwork.

sorghr(a,tau,[lo,hi,lwork,overwrite_a])

dorghr(a,tau,[lo,hi,lwork,overwrite_a])

sorghr_lwork(n,[lo,hi])

Wrapper for sorghr_lwork.

dorghr_lwork(n,[lo,hi])

Wrapper for dorghr_lwork.

sorgqr(a,tau,[lwork,overwrite_a])

dorgqr(a,tau,[lwork,overwrite_a])

sorgrq(a,tau,[lwork,overwrite_a])

dorgrq(a,tau,[lwork,overwrite_a])

sormqr(side,trans,a,tau,c,lwork,[overwrite_c])

dormqr(side,trans,a,tau,c,lwork,[overwrite_c])

sormrz(a,tau,c,[side,trans,lwork,overwrite_c])

dormrz(a,tau,c,[side,trans,lwork,overwrite_c])

sormrz_lwork(m,n,[side,trans])

Wrapper for sormrz_lwork.

dormrz_lwork(m,n,[side,trans])

Wrapper for dormrz_lwork.

spbsv(ab,b,[lower,ldab,overwrite_ab,overwrite_b])

dpbsv(ab,b,[lower,ldab,overwrite_ab,overwrite_b])

cpbsv(ab,b,[lower,ldab,overwrite_ab,overwrite_b])

zpbsv(ab,b,[lower,ldab,overwrite_ab,overwrite_b])

spbtrf(ab,[lower,ldab,overwrite_ab])

dpbtrf(ab,[lower,ldab,overwrite_ab])

cpbtrf(ab,[lower,ldab,overwrite_ab])

zpbtrf(ab,[lower,ldab,overwrite_ab])

spbtrs(ab,b,[lower,ldab,overwrite_b])

dpbtrs(ab,b,[lower,ldab,overwrite_b])

cpbtrs(ab,b,[lower,ldab,overwrite_b])

zpbtrs(ab,b,[lower,ldab,overwrite_b])

spftrf(n,a,[transr,uplo,overwrite_a])

dpftrf(n,a,[transr,uplo,overwrite_a])

cpftrf(n,a,[transr,uplo,overwrite_a])

zpftrf(n,a,[transr,uplo,overwrite_a])

spftri(n,a,[transr,uplo,overwrite_a])

dpftri(n,a,[transr,uplo,overwrite_a])

cpftri(n,a,[transr,uplo,overwrite_a])

zpftri(n,a,[transr,uplo,overwrite_a])

spftrs(n,a,b,[transr,uplo,overwrite_b])

dpftrs(n,a,b,[transr,uplo,overwrite_b])

cpftrs(n,a,b,[transr,uplo,overwrite_b])

zpftrs(n,a,b,[transr,uplo,overwrite_b])

spocon(a,anorm,[uplo])

dpocon(a,anorm,[uplo])

cpocon(a,anorm,[uplo])

zpocon(a,anorm,[uplo])

spstrf(a,[tol,lower,overwrite_a])

dpstrf(a,[tol,lower,overwrite_a])

cpstrf(a,[tol,lower,overwrite_a])

zpstrf(a,[tol,lower,overwrite_a])

spstf2(a,[tol,lower,overwrite_a])

dpstf2(a,[tol,lower,overwrite_a])

cpstf2(a,[tol,lower,overwrite_a])

zpstf2(a,[tol,lower,overwrite_a])

sposv(a,b,[lower,overwrite_a,overwrite_b])

dposv(a,b,[lower,overwrite_a,overwrite_b])

cposv(a,b,[lower,overwrite_a,overwrite_b])

zposv(a,b,[lower,overwrite_a,overwrite_b])

spotrf(a,[lower,clean,overwrite_a])

dpotrf(a,[lower,clean,overwrite_a])

cpotrf(a,[lower,clean,overwrite_a])

zpotrf(a,[lower,clean,overwrite_a])

spotri(c,[lower,overwrite_c])

dpotri(c,[lower,overwrite_c])

cpotri(c,[lower,overwrite_c])

zpotri(c,[lower,overwrite_c])

spotrs(c,b,[lower,overwrite_b])

dpotrs(c,b,[lower,overwrite_b])

cpotrs(c,b,[lower,overwrite_b])

zpotrs(c,b,[lower,overwrite_b])

sppcon(n,ap,anorm,[lower])

dppcon(n,ap,anorm,[lower])

cppcon(n,ap,anorm,[lower])

zppcon(n,ap,anorm,[lower])

sppsv(n,ap,b,[lower,overwrite_b])

dppsv(n,ap,b,[lower,overwrite_b])

cppsv(n,ap,b,[lower,overwrite_b])

zppsv(n,ap,b,[lower,overwrite_b])

spptrf(n,ap,[lower,overwrite_ap])

dpptrf(n,ap,[lower,overwrite_ap])

cpptrf(n,ap,[lower,overwrite_ap])

zpptrf(n,ap,[lower,overwrite_ap])

spptri(n,ap,[lower,overwrite_ap])

dpptri(n,ap,[lower,overwrite_ap])

cpptri(n,ap,[lower,overwrite_ap])

zpptri(n,ap,[lower,overwrite_ap])

spptrs(n,ap,b,[lower,overwrite_b])

dpptrs(n,ap,b,[lower,overwrite_b])

cpptrs(n,ap,b,[lower,overwrite_b])

zpptrs(n,ap,b,[lower,overwrite_b])

sptsvx(d,e,b,[fact,df,ef])

dptsvx(d,e,b,[fact,df,ef])

cptsvx(d,e,b,[fact,df,ef])

zptsvx(d,e,b,[fact,df,ef])

spttrf(d,e,[overwrite_d,overwrite_e])

dpttrf(d,e,[overwrite_d,overwrite_e])

cpttrf(d,e,[overwrite_d,overwrite_e])

zpttrf(d,e,[overwrite_d,overwrite_e])

spttrs(d,e,b,[overwrite_b])

dpttrs(d,e,b,[overwrite_b])

cpttrs(d,e,b,[lower,overwrite_b])

zpttrs(d,e,b,[lower,overwrite_b])

ssbev(ab,[compute_v,lower,ldab,overwrite_ab])

dsbev(ab,[compute_v,lower,ldab,overwrite_ab])

sstebz(d, e, range, vl, vu, il, iu, tol, order)

dstebz(d, e, range, vl, vu, il, iu, tol, order)

sstein(d, e, w, iblock, isplit)

dstein(d, e, w, iblock, isplit)

Wrapper for sstemr_lwork.

Wrapper for dstemr_lwork.

ssterf(d,e,[overwrite_d,overwrite_e])

dsterf(d,e,[overwrite_d,overwrite_e])

sstev(d,e,[compute_v,overwrite_d,overwrite_e])

dstev(d,e,[compute_v,overwrite_d,overwrite_e])

ssycon(a,ipiv,anorm,[lower])

dsycon(a,ipiv,anorm,[lower])

csycon(a,ipiv,anorm,[lower])

zsycon(a,ipiv,anorm,[lower])

ssyconv(a,ipiv,[lower,way,overwrite_a])

dsyconv(a,ipiv,[lower,way,overwrite_a])

csyconv(a,ipiv,[lower,way,overwrite_a])

zsyconv(a,ipiv,[lower,way,overwrite_a])

ssyev(a,[compute_v,lower,lwork,overwrite_a])

dsyev(a,[compute_v,lower,lwork,overwrite_a])

ssyev_lwork(n,[lower])

Wrapper for ssyev_lwork.

dsyev_lwork(n,[lower])

Wrapper for dsyev_lwork.

ssyevd_lwork(n,[compute_v,lower])

Wrapper for ssyevd_lwork.

dsyevd_lwork(n,[compute_v,lower])

Wrapper for dsyevd_lwork.

ssyevr_lwork(n,[lower])

Wrapper for ssyevr_lwork.

dsyevr_lwork(n,[lower])

Wrapper for dsyevr_lwork.

ssyevx_lwork(n,[lower])

Wrapper for ssyevx_lwork.

dsyevx_lwork(n,[lower])

Wrapper for dsyevx_lwork.

ssygst(a,b,[itype,lower,overwrite_a])

dsygst(a,b,[itype,lower,overwrite_a])

ssygv_lwork(n,[uplo])

Wrapper for ssygv_lwork.

dsygv_lwork(n,[uplo])

Wrapper for dsygv_lwork.

ssygvx_lwork(n,[uplo])

Wrapper for ssygvx_lwork.

dsygvx_lwork(n,[uplo])

Wrapper for dsygvx_lwork.

ssysv(a,b,[lwork,lower,overwrite_a,overwrite_b])

dsysv(a,b,[lwork,lower,overwrite_a,overwrite_b])

csysv(a,b,[lwork,lower,overwrite_a,overwrite_b])

zsysv(a,b,[lwork,lower,overwrite_a,overwrite_b])

ssysv_lwork(n,[lower])

Wrapper for ssysv_lwork.

dsysv_lwork(n,[lower])

Wrapper for dsysv_lwork.

csysv_lwork(n,[lower])

Wrapper for csysv_lwork.

zsysv_lwork(n,[lower])

Wrapper for zsysv_lwork.

ssysvx_lwork(n,[lower])

Wrapper for ssysvx_lwork.

dsysvx_lwork(n,[lower])

Wrapper for dsysvx_lwork.

csysvx_lwork(n,[lower])

Wrapper for csysvx_lwork.

zsysvx_lwork(n,[lower])

Wrapper for zsysvx_lwork.

ssytf2(a,[lower,overwrite_a])

dsytf2(a,[lower,overwrite_a])

csytf2(a,[lower,overwrite_a])

zsytf2(a,[lower,overwrite_a])

ssytrd(a,[lower,lwork,overwrite_a])

dsytrd(a,[lower,lwork,overwrite_a])

ssytrd_lwork(n,[lower])

Wrapper for ssytrd_lwork.

dsytrd_lwork(n,[lower])

Wrapper for dsytrd_lwork.

ssytrf(a,[lower,lwork,overwrite_a])

dsytrf(a,[lower,lwork,overwrite_a])

csytrf(a,[lower,lwork,overwrite_a])

zsytrf(a,[lower,lwork,overwrite_a])

ssytrf_lwork(n,[lower])

Wrapper for ssytrf_lwork.

dsytrf_lwork(n,[lower])

Wrapper for dsytrf_lwork.

csytrf_lwork(n,[lower])

Wrapper for csytrf_lwork.

zsytrf_lwork(n,[lower])

Wrapper for zsytrf_lwork.

ssytri(a,ipiv,[lower,overwrite_a])

dsytri(a,ipiv,[lower,overwrite_a])

csytri(a,ipiv,[lower,overwrite_a])

zsytri(a,ipiv,[lower,overwrite_a])

ssytrs(a,ipiv,b,[lower,overwrite_b])

dsytrs(a,ipiv,b,[lower,overwrite_b])

csytrs(a,ipiv,b,[lower,overwrite_b])

zsytrs(a,ipiv,b,[lower,overwrite_b])

stbtrs(ab,b,[uplo,trans,diag,overwrite_b])

dtbtrs(ab,b,[uplo,trans,diag,overwrite_b])

ctbtrs(ab,b,[uplo,trans,diag,overwrite_b])

ztbtrs(ab,b,[uplo,trans,diag,overwrite_b])

stfttp(n,arf,[transr,uplo])

dtfttp(n,arf,[transr,uplo])

ctfttp(n,arf,[transr,uplo])

ztfttp(n,arf,[transr,uplo])

stfttr(n,arf,[transr,uplo])

dtfttr(n,arf,[transr,uplo])

ctfttr(n,arf,[transr,uplo])

ztfttr(n,arf,[transr,uplo])

stgsen_lwork(select,a,[ijob])

Wrapper for stgsen_lwork.

dtgsen_lwork(select,a,[ijob])

Wrapper for dtgsen_lwork.

ctgsen_lwork(select,a,b,[ijob])

Wrapper for ctgsen_lwork.

ztgsen_lwork(select,a,b,[ijob])

Wrapper for ztgsen_lwork.

stpttf(n,ap,[transr,uplo])

dtpttf(n,ap,[transr,uplo])

ctpttf(n,ap,[transr,uplo])

ztpttf(n,ap,[transr,uplo])

strcon(a,[norm,uplo,diag])

dtrcon(a,[norm,uplo,diag])

ctrcon(a,[norm,uplo,diag])

ztrcon(a,[norm,uplo,diag])

strsen_lwork(select,t,[job])

Wrapper for strsen_lwork.

dtrsen_lwork(select,t,[job])

Wrapper for dtrsen_lwork.

ctrsen_lwork(select,t,[job])

Wrapper for ctrsen_lwork.

ztrsen_lwork(select,t,[job])

Wrapper for ztrsen_lwork.

strsyl(a,b,c,[trana,tranb,isgn,overwrite_c])

dtrsyl(a,b,c,[trana,tranb,isgn,overwrite_c])

ctrsyl(a,b,c,[trana,tranb,isgn,overwrite_c])

ztrsyl(a,b,c,[trana,tranb,isgn,overwrite_c])

strtri(c,[lower,unitdiag,overwrite_c])

dtrtri(c,[lower,unitdiag,overwrite_c])

ctrtri(c,[lower,unitdiag,overwrite_c])

ztrtri(c,[lower,unitdiag,overwrite_c])

strttf(a,[transr,uplo])

dtrttf(a,[transr,uplo])

ctrttf(a,[transr,uplo])

ztrttf(a,[transr,uplo])

stzrzf(a,[lwork,overwrite_a])

dtzrzf(a,[lwork,overwrite_a])

ctzrzf(a,[lwork,overwrite_a])

ztzrzf(a,[lwork,overwrite_a])

Wrapper for stzrzf_lwork.

Wrapper for dtzrzf_lwork.

Wrapper for ctzrzf_lwork.

Wrapper for ztzrzf_lwork.

cunghr(a,tau,[lo,hi,lwork,overwrite_a])

zunghr(a,tau,[lo,hi,lwork,overwrite_a])

cunghr_lwork(n,[lo,hi])

Wrapper for cunghr_lwork.

zunghr_lwork(n,[lo,hi])

Wrapper for zunghr_lwork.

cungqr(a,tau,[lwork,overwrite_a])

zungqr(a,tau,[lwork,overwrite_a])

cungrq(a,tau,[lwork,overwrite_a])

zungrq(a,tau,[lwork,overwrite_a])

cunmqr(side,trans,a,tau,c,lwork,[overwrite_c])

zunmqr(side,trans,a,tau,c,lwork,[overwrite_c])

sgeqrt(nb,a,[overwrite_a])

dgeqrt(nb,a,[overwrite_a])

cgeqrt(nb,a,[overwrite_a])

zgeqrt(nb,a,[overwrite_a])

sgemqrt(v,t,c,[side,trans,overwrite_c])

dgemqrt(v,t,c,[side,trans,overwrite_c])

cgemqrt(v,t,c,[side,trans,overwrite_c])

zgemqrt(v,t,c,[side,trans,overwrite_c])

sgttrs(dl,d,du,du2,ipiv,b,[trans,overwrite_b])

dgttrs(dl,d,du,du2,ipiv,b,[trans,overwrite_b])

cgttrs(dl,d,du,du2,ipiv,b,[trans,overwrite_b])

zgttrs(dl,d,du,du2,ipiv,b,[trans,overwrite_b])

sgtcon(dl,d,du,du2,ipiv,anorm,[norm])

dgtcon(dl,d,du,du2,ipiv,anorm,[norm])

cgtcon(dl,d,du,du2,ipiv,anorm,[norm])

zgtcon(dl,d,du,du2,ipiv,anorm,[norm])

stpqrt(l,nb,a,b,[overwrite_a,overwrite_b])

dtpqrt(l,nb,a,b,[overwrite_a,overwrite_b])

ctpqrt(l,nb,a,b,[overwrite_a,overwrite_b])

ztpqrt(l,nb,a,b,[overwrite_a,overwrite_b])

cuncsd_lwork(m, p, q)

Wrapper for cuncsd_lwork.

zuncsd_lwork(m, p, q)

Wrapper for zuncsd_lwork.

cunmrz(a,tau,c,[side,trans,lwork,overwrite_c])

zunmrz(a,tau,c,[side,trans,lwork,overwrite_c])

cunmrz_lwork(m,n,[side,trans])

Wrapper for cunmrz_lwork.

zunmrz_lwork(m,n,[side,trans])

Wrapper for zunmrz_lwork.

---

## rank_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rank_filter.html

**Contents:**
- rank_filter#

Calculate a multidimensional rank filter.

The rank parameter may be less than zero, i.e., rank = -1 indicates the largest element.

See footprint, below. Ignored if footprint is given.

Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function. footprint is a boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)). We adjust size to the number of dimensions of the input array, so that, if the input array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When footprint is given, size is ignored.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for size, origin, and/or mode must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

Filtered array. Has the same shape as input.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = ndimage.rank_filter(ascent, rank=42, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## det#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.det.html

**Contents:**
- det#

Compute the determinant of a matrix

The determinant is a scalar that is a function of the associated square matrix coefficients. The determinant value is zero for singular matrices.

Input array to compute determinants for.

Allow overwriting data in a (may enhance performance).

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Determinant of a. For stacked arrays, a scalar is returned for each (m, m) slice in the last two dimensions of the input. For example, an input of shape (p, q, m, m) will produce a result of shape (p, q). If all dimensions are 1 a scalar is returned regardless of ndim.

The determinant is computed by performing an LU factorization of the input with LAPACK routine ‘getrf’, and then calculating the product of diagonal entries of the U factor.

Even if the input array is single precision (float32 or complex64), the result will be returned in double precision (float64 or complex128) to prevent overflows.

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> a = np.array([[1,2,3], [4,5,6], [7,8,9]])  # A singular matrix
>>> linalg.det(a)
0.0
>>> b = np.array([[0,2,3], [4,5,6], [7,8,9]])
>>> linalg.det(b)
3.0
>>> # An array with the shape (3, 2, 2, 2)
>>> c = np.array([[[[1., 2.], [3., 4.]],
...                [[5., 6.], [7., 8.]]],
...               [[[9., 10.], [11., 12.]],
...                [[13., 14.], [15., 16.]]],
...               [[[17., 18.], [19., 20.]],
...                [[21., 22.], [23., 24.]]]])
>>> linalg.det(c)  # The resulting shape is (3, 2)
array([[-2., -2.],
       [-2., -2.],
       [-2., -2.]])
>>> linalg.det(c[0, 0])  # Confirm the (0, 0) slice, [[1, 2], [3, 4]]
-2.0
```

---

## rfft2#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft2.html

**Contents:**
- rfft2#

Compute the 2-D FFT of a real array.

Input array, taken to be real.

Axes over which to compute the FFT.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The result of the real 2-D FFT.

The inverse of the 2-D FFT of real input.

The 1-D FFT of real input.

Compute the N-D discrete Fourier Transform for real input.

This is really just rfftn with different default behavior. For more details see rfftn.

**Examples:**

Example 1 (json):
```json
>>> import scipy.fft
>>> import numpy as np
>>> x = np.broadcast_to([1, 0, -1, 0], (4, 4))
>>> scipy.fft.rfft2(x)
array([[0.+0.j, 8.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j]])
```

---

## savemat#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html

**Contents:**
- savemat#

Save a dictionary of names and arrays into a MATLAB-style .mat file.

This saves the array objects in the given dictionary to a MATLAB- style .mat file.

Name of the .mat file (.mat extension not needed if appendmat == True). Can also pass open file_like object.

Dictionary from which to save matfile variables. Note that if this dict has a key starting with _ or a sub-dict has a key starting with _ or a digit, these key’s items will not be saved in the mat file and MatWriteWarning will be issued.

True (the default) to append the .mat extension to the end of the given filename, if not already present.

‘5’ (the default) for MATLAB 5 and up (to 7.2), ‘4’ for MATLAB 4 .mat files.

False (the default) - maximum field name length in a structure is 31 characters which is the documented maximum length. True - maximum field name length in a structure is 63 characters which works for MATLAB 7.6+.

Whether or not to compress matrices on write. Default is False.

If ‘column’, write 1-D NumPy arrays as column vectors. If ‘row’, write 1-D NumPy arrays as row vectors.

**Examples:**

Example 1 (json):
```json
>>> from scipy.io import savemat
>>> import numpy as np
>>> a = np.arange(20)
>>> mdic = {"a": a, "label": "experiment"}
>>> mdic
{'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19]),
'label': 'experiment'}
>>> savemat("matlab_matrix.mat", mdic)
```

---

## convolve1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html

**Contents:**
- convolve1d#

Calculate a 1-D convolution along the given axis.

The lines of the array along the given axis are convolved with the given weights.

1-D sequence of numbers.

The axis of input along which to calculate. Default is -1.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right.

Convolved array with same shape as input

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.ndimage import convolve1d
>>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
array([14, 24,  4, 13, 12, 36, 27,  0])
```

---

## eigvals#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvals.html

**Contents:**
- eigvals#

Compute eigenvalues from an ordinary or generalized eigenvalue problem.

Find eigenvalues of a general matrix:

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

A complex or real matrix whose eigenvalues and eigenvectors will be computed.

Right-hand side matrix in a generalized eigenvalue problem. If omitted, identity matrix is assumed.

Whether to overwrite data in a (may improve performance)

Whether to check that the input matrices contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

If True, return the eigenvalues in homogeneous coordinates. In this case w is a (2, M) array so that:

The eigenvalues, each repeated according to its multiplicity but not in any specific order. The shape is (M,) unless homogeneous_eigvals=True.

If eigenvalue computation does not converge

eigenvalues and right eigenvectors of general arrays.

eigenvalues of symmetric or Hermitian arrays

eigenvalues for symmetric/Hermitian band matrices

eigenvalues of symmetric/Hermitian tridiagonal matrices

**Examples:**

Example 1 (unknown):
```unknown
a   vr[:,i] = w[i]        b   vr[:,i]
```

Example 2 (unknown):
```unknown
w[1,i] a vr[:,i] = w[0,i] b vr[:,i]
```

Example 3 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> a = np.array([[0., -1.], [1., 0.]])
>>> linalg.eigvals(a)
array([0.+1.j, 0.-1.j])
```

Example 4 (unknown):
```unknown
>>> b = np.array([[0., 1.], [1., 1.]])
>>> linalg.eigvals(a, b)
array([ 1.+0.j, -1.+0.j])
```

---

## maximum_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter.html

**Contents:**
- maximum_filter#

Calculate a multidimensional maximum filter.

See footprint, below. Ignored if footprint is given.

Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function. footprint is a boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)). We adjust size to the number of dimensions of the input array, so that, if the input array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When footprint is given, size is ignored.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for size, origin, and/or mode must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

Filtered array. Has the same shape as input.

A sequence of modes (one per axis) is only supported when the footprint is separable. Otherwise, a single mode string must be provided.

The behavior of this function with NaN elements is undefined. To control behavior in the presence of NaNs, consider using vectorized_filter.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = ndimage.maximum_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## make_smoothing_spline#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html

**Contents:**
- make_smoothing_spline#

Create a smoothing B-spline satisfying the Generalized Cross Validation (GCV) criterion.

Compute the (coefficients of) smoothing cubic spline function using lam to control the tradeoff between the amount of smoothness of the curve and its proximity to the data. In case lam is None, using the GCV criteria [1] to find it.

A smoothing spline is found as a solution to the regularized weighted linear regression problem:

where \(f\) is a spline function, \(w\) is a vector of weights and \(\lambda\) is a regularization parameter.

If lam is None, we use the GCV criteria to find an optimal regularization parameter, otherwise we solve the regularized weighted linear regression problem with given parameter. The parameter controls the tradeoff in the following way: the larger the parameter becomes, the smoother the function gets.

Abscissas. n must be at least 5.

Ordinates. n must be at least 5.

Vector of weights. Default is np.ones_like(x).

Regularization parameter. If lam is None, then it is found from the GCV criteria. Default is None.

The data axis. Default is zero. The assumption is that y.shape[axis] == n, and all other axes of y are batching axes.

An object representing a spline in the B-spline basis as a solution of the problem of smoothing splines using the GCV criteria [1] in case lam is None, otherwise using the given parameter lam.

This algorithm is a clean room reimplementation of the algorithm introduced by Woltring in FORTRAN [2]. The original version cannot be used in SciPy source code because of the license issues. The details of the reimplementation are discussed here (available only in Russian) [4].

If the vector of weights w is None, we assume that all the points are equal in terms of weights, and vector of weights is vector of ones.

Note that in weighted residual sum of squares, weights are not squared: \(\sum\limits_{i=1}^n w_i\lvert y_i - f(x_i) \rvert^2\) while in splrep the sum is built from the squared weights.

In cases when the initial problem is ill-posed (for example, the product \(X^T W X\) where \(X\) is a design matrix is not a positive defined matrix) a ValueError is raised.

G. Wahba, “Estimating the smoothing parameter” in Spline models for observational data, Philadelphia, Pennsylvania: Society for Industrial and Applied Mathematics, 1990, pp. 45-65. DOI:10.1137/1.9781611970128

H. J. Woltring, A Fortran package for generalized, cross-validatory spline smoothing and differentiation, Advances in Engineering Software, vol. 8, no. 2, pp. 104-113, 1986. DOI:10.1016/0141-1195(86)90098-7

T. Hastie, J. Friedman, and R. Tisbshirani, “Smoothing Splines” in The elements of Statistical Learning: Data Mining, Inference, and prediction, New York: Springer, 2017, pp. 241-249. DOI:10.1007/978-0-387-84858-7

E. Zemlyanoy, “Generalized cross-validation smoothing splines”, BSc thesis, 2022. https://www.hse.ru/ba/am/students/diplomas/620910604 (in Russian)

Generate some noisy data

Make a smoothing spline function

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> np.random.seed(1234)
>>> n = 200
>>> def func(x):
...    return x**3 + x**2 * np.sin(4 * x)
>>> x = np.sort(np.random.random_sample(n) * 4 - 2)
>>> y = func(x) + np.random.normal(scale=1.5, size=n)
```

Example 2 (sql):
```sql
>>> from scipy.interpolate import make_smoothing_spline
>>> spl = make_smoothing_spline(x, y)
```

Example 3 (typescript):
```typescript
>>> import matplotlib.pyplot as plt
>>> grid = np.linspace(x[0], x[-1], 400)
>>> plt.plot(x, y, '.')
>>> plt.plot(grid, spl(grid), label='Spline')
>>> plt.plot(grid, func(grid), label='Original function')
>>> plt.legend(loc='best')
>>> plt.show()
```

---

## Signal Processing (scipy.signal)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/signal.html

**Contents:**
- Signal Processing (scipy.signal)#
- B-splines#
- Filtering#
  - Convolution/Correlation#
  - Difference-equation filtering#
    - Analysis of Linear Systems#
  - Filter Design#
    - FIR Filter#
    - IIR Filter#
    - Filter Coefficients#

The signal processing toolbox currently contains some filtering functions, a limited set of filter design tools, and a few B-spline interpolation algorithms for 1- and 2-D data. While the B-spline algorithms could technically be placed under the interpolation category, they are included here because they only work with equally-spaced data and make heavy use of filter-theory and transfer-function formalism to provide a fast B-spline transform. To understand this section, you will need to understand that a signal in SciPy is an array of real or complex numbers.

A B-spline is an approximation of a continuous function over a finite- domain in terms of B-spline coefficients and knot points. If the knot- points are equally spaced with spacing \(\Delta x\), then the B-spline approximation to a 1-D function is the finite-basis expansion.

In two dimensions with knot-spacing \(\Delta x\) and \(\Delta y\), the function representation is

In these expressions, \(\beta^{o}\left(\cdot\right)\) is the space-limited B-spline basis function of order \(o\). The requirement of equally-spaced knot-points and equally-spaced data points, allows the development of fast (inverse-filtering) algorithms for determining the coefficients, \(c_{j}\), from sample-values, \(y_{n}\). Unlike the general spline interpolation algorithms, these algorithms can quickly find the spline coefficients for large images.

The advantage of representing a set of samples via B-spline basis functions is that continuous-domain operators (derivatives, re- sampling, integral, etc.), which assume that the data samples are drawn from an underlying continuous function, can be computed with relative ease from the spline coefficients. For example, the second derivative of a spline is

Using the property of B-splines that

If \(o=3\), then at the sample points:

Thus, the second-derivative signal can be easily calculated from the spline fit. If desired, smoothing splines can be found to make the second derivative less sensitive to random errors.

The savvy reader will have already noticed that the data samples are related to the knot coefficients via a convolution operator, so that simple convolution with the sampled B-spline function recovers the original data from the spline coefficients. The output of convolutions can change depending on how the boundaries are handled (this becomes increasingly more important as the number of dimensions in the dataset increases). The algorithms relating to B-splines in the signal-processing subpackage assume mirror-symmetric boundary conditions. Thus, spline coefficients are computed based on that assumption, and data-samples can be recovered exactly from the spline coefficients by assuming them to be mirror-symmetric also.

Currently the package provides functions for determining second- and third- order cubic spline coefficients from equally-spaced samples in one and two dimensions (qspline1d, qspline2d, cspline1d, cspline2d). For large \(o\), the B-spline basis function can be approximated well by a zero-mean Gaussian function with standard-deviation equal to \(\sigma_{o}=\left(o+1\right)/12\) :

A function to compute this Gaussian for arbitrary \(x\) and \(o\) is also available ( gauss_spline ). The following code and figure use spline-filtering to compute an edge-image (the second derivative of a smoothed spline) of a raccoon’s face, which is an array returned by the command scipy.datasets.face. The command sepfir2d was used to apply a separable 2-D FIR filter with mirror-symmetric boundary conditions to the spline coefficients. This function is ideally-suited for reconstructing samples from spline coefficients and is faster than convolve2d, which convolves arbitrary 2-D filters and allows for choosing mirror-symmetric boundary conditions.

Alternatively, we could have done:

Filtering is a generic name for any system that modifies an input signal in some way. In SciPy, a signal can be thought of as a NumPy array. There are different kinds of filters for different kinds of operations. There are two broad kinds of filtering operations: linear and non-linear. Linear filters can always be reduced to multiplication of the flattened NumPy array by an appropriate matrix resulting in another flattened NumPy array. Of course, this is not usually the best way to compute the filter, as the matrices and vectors involved may be huge. For example, filtering a \(512 \times 512\) image with this method would require multiplication of a \(512^2 \times 512^2\) matrix with a \(512^2\) vector. Just trying to store the \(512^2 \times 512^2\) matrix using a standard NumPy array would require \(68,719,476,736\) elements. At 4 bytes per element this would require \(256\textrm{GB}\) of memory. In most applications, most of the elements of this matrix are zero and a different method for computing the output of the filter is employed.

Many linear filters also have the property of shift-invariance. This means that the filtering operation is the same at different locations in the signal and it implies that the filtering matrix can be constructed from knowledge of one row (or column) of the matrix alone. In this case, the matrix multiplication can be accomplished using Fourier transforms.

Let \(x\left[n\right]\) define a 1-D signal indexed by the integer \(n.\) Full convolution of two 1-D signals can be expressed as

This equation can only be implemented directly if we limit the sequences to finite-support sequences that can be stored in a computer, choose \(n=0\) to be the starting point of both sequences, let \(K+1\) be that value for which \(x\left[n\right]=0\) for all \(n\geq K+1\) and \(M+1\) be that value for which \(h\left[n\right]=0\) for all \(n\geq M+1\), then the discrete convolution expression is

For convenience, assume \(K\geq M.\) Then, more explicitly, the output of this operation is

Thus, the full discrete convolution of two finite sequences of lengths \(K+1\) and \(M+1\), respectively, results in a finite sequence of length \(K+M+1=\left(K+1\right)+\left(M+1\right)-1.\)

1-D convolution is implemented in SciPy with the function convolve. This function takes as inputs the signals \(x,\) \(h\), and two optional flags ‘mode’ and ‘method’, and returns the signal \(y.\)

The first optional flag, ‘mode’, allows for the specification of which part of the output signal to return. The default value of ‘full’ returns the entire signal. If the flag has a value of ‘same’, then only the middle \(K\) values are returned, starting at \(y\left[\left\lfloor \frac{M-1}{2}\right\rfloor \right]\), so that the output has the same length as the first input. If the flag has a value of ‘valid’, then only the middle \(K-M+1=\left(K+1\right)-\left(M+1\right)+1\) output values are returned, where \(z\) depends on all of the values of the smallest input from \(h\left[0\right]\) to \(h\left[M\right].\) In other words, only the values \(y\left[M\right]\) to \(y\left[K\right]\) inclusive are returned.

The second optional flag, ‘method’, determines how the convolution is computed, either through the Fourier transform approach with fftconvolve or through the direct method. By default, it selects the expected faster method. The Fourier transform method has order \(O(N\log N)\), while the direct method has order \(O(N^2)\). Depending on the big O constant and the value of \(N\), one of these two methods may be faster. The default value, ‘auto’, performs a rough calculation and chooses the expected faster method, while the values ‘direct’ and ‘fft’ force computation with the other two methods.

The code below shows a simple example for convolution of 2 sequences:

This same function convolve can actually take N-D arrays as inputs and will return the N-D convolution of the two arrays, as is shown in the code example below. The same input flags are available for that case as well.

Correlation is very similar to convolution except that the minus sign becomes a plus sign. Thus,

is the (cross) correlation of the signals \(y\) and \(x.\) For finite-length signals with \(y\left[n\right]=0\) outside of the range \(\left[0,K\right]\) and \(x\left[n\right]=0\) outside of the range \(\left[0,M\right],\) the summation can simplify to

Assuming again that \(K\geq M\), this is

The SciPy function correlate implements this operation. Equivalent flags are available for this operation to return the full \(K+M+1\) length sequence (‘full’) or a sequence with the same size as the largest sequence starting at \(w\left[-K+\left\lfloor \frac{M-1}{2}\right\rfloor \right]\) (‘same’) or a sequence where the values depend on all the values of the smallest sequence (‘valid’). This final option returns the \(K-M+1\) values \(w\left[M-K\right]\) to \(w\left[0\right]\) inclusive.

The function correlate can also take arbitrary N-D arrays as input and return the N-D convolution of the two arrays on output.

When \(N=2,\) correlate and/or convolve can be used to construct arbitrary image filters to perform actions such as blurring, enhancing, and edge-detection for an image.

Calculating the convolution in the time domain as above is mainly used for filtering when one of the signals is much smaller than the other ( \(K\gg M\) ), otherwise linear filtering is more efficiently calculated in the frequency domain provided by the function fftconvolve. By default, convolve estimates the fastest method using choose_conv_method.

If the filter function \(w[n,m]\) can be factored according to

convolution can be calculated by means of the function sepfir2d. As an example, we consider a Gaussian filter gaussian

which is often used for blurring.

A general class of linear 1-D filters (that includes convolution filters) are filters described by the difference equation

where \(x\left[n\right]\) is the input sequence and \(y\left[n\right]\) is the output sequence. If we assume initial rest so that \(y\left[n\right]=0\) for \(n<0\), then this kind of filter can be implemented using convolution. However, the convolution filter sequence \(h\left[n\right]\) could be infinite if \(a_{k}\neq0\) for \(k\geq1.\) In addition, this general class of linear filter allows initial conditions to be placed on \(y\left[n\right]\) for \(n<0\) resulting in a filter that cannot be expressed using convolution.

The difference equation filter can be thought of as finding \(y\left[n\right]\) recursively in terms of its previous values

Often, \(a_{0}=1\) is chosen for normalization. The implementation in SciPy of this general difference equation filter is a little more complicated than would be implied by the previous equation. It is implemented so that only one signal needs to be delayed. The actual implementation equations are (assuming \(a_{0}=1\) ):

where \(K=\max\left(N,M\right).\) Note that \(b_{K}=0\) if \(K>M\) and \(a_{K}=0\) if \(K>N.\) In this way, the output at time \(n\) depends only on the input at time \(n\) and the value of \(z_{0}\) at the previous time. This can always be calculated as long as the \(K\) values \(z_{0}\left[n-1\right]\ldots z_{K-1}\left[n-1\right]\) are computed and stored at each time step.

The difference-equation filter is called using the command lfilter in SciPy. This command takes as inputs the vector \(b,\) the vector, \(a,\) a signal \(x\) and returns the vector \(y\) (the same length as \(x\) ) computed using the equation given above. If \(x\) is N-D, then the filter is computed along the axis provided. If desired, initial conditions providing the values of \(z_{0}\left[-1\right]\) to \(z_{K-1}\left[-1\right]\) can be provided or else it will be assumed that they are all zero. If initial conditions are provided, then the final conditions on the intermediate variables are also returned. These could be used, for example, to restart the calculation in the same state.

Sometimes, it is more convenient to express the initial conditions in terms of the signals \(x\left[n\right]\) and \(y\left[n\right].\) In other words, perhaps you have the values of \(x\left[-M\right]\) to \(x\left[-1\right]\) and the values of \(y\left[-N\right]\) to \(y\left[-1\right]\) and would like to determine what values of \(z_{m}\left[-1\right]\) should be delivered as initial conditions to the difference-equation filter. It is not difficult to show that, for \(0\leq m<K,\)

Using this formula, we can find the initial-condition vector \(z_{0}\left[-1\right]\) to \(z_{K-1}\left[-1\right]\) given initial conditions on \(y\) (and \(x\) ). The command lfiltic performs this function.

As an example, consider the following system:

The code calculates the signal \(y[n]\) for a given signal \(x[n]\); first for initial conditions \(y[-1] = 0\) (default case), then for \(y[-1] = 2\) by means of lfiltic.

Note that the output signal \(y[n]\) has the same length as the length as the input signal \(x[n]\).

Linear system described a linear-difference equation can be fully described by the coefficient vectors \(a\) and \(b\) as was done above; an alternative representation is to provide a factor \(k\), \(N_z\) zeros \(z_k\) and \(N_p\) poles \(p_k\), respectively, to describe the system by means of its transfer function \(H(z)\), according to

This alternative representation can be obtained with the scipy function tf2zpk; the inverse is provided by zpk2tf.

For the above example we have

i.e., the system has a zero at \(z=-1/2\) and a pole at \(z=1/3\).

The scipy function freqz allows calculation of the frequency response of a system described by the coefficients \(a_k\) and \(b_k\). See the help of the freqz function for a comprehensive example.

Time-discrete filters can be classified into finite response (FIR) filters and infinite response (IIR) filters. FIR filters can provide a linear phase response, whereas IIR filters cannot. SciPy provides functions for designing both types of filters.

The function firwin designs filters according to the window method. Depending on the provided arguments, the function returns different filter types (e.g., low-pass, band-pass…).

The example below designs a low-pass and a band-stop filter, respectively.

Note that firwin uses, per default, a normalized frequency defined such that the value \(1\) corresponds to the Nyquist frequency, whereas the function freqz is defined such that the value \(\pi\) corresponds to the Nyquist frequency.

The function firwin2 allows design of almost arbitrary frequency responses by specifying an array of corner frequencies and corresponding gains, respectively.

The example below designs a filter with such an arbitrary amplitude response.

Note the linear scaling of the y-axis and the different definition of the Nyquist frequency in firwin2 and freqz (as explained above).

SciPy provides two functions to directly design IIR iirdesign and iirfilter, where the filter type (e.g., elliptic) is passed as an argument and several more filter design functions for specific filter types, e.g., ellip.

The example below designs an elliptic low-pass filter with defined pass-band and stop-band ripple, respectively. Note the much lower filter order (order 4) compared with the FIR filters from the examples above in order to reach the same stop-band attenuation of \(\approx 60\) dB.

It is important to note that the cutoffs for firwin and iirfilter are defined differently. For firwin, the cutoff-frequency is at half-amplitude (i.e. -6dB). For iirfilter, the cutoff is at half-power (i.e. -3dB).

Filter coefficients can be stored in several different formats:

‘ba’ or ‘tf’ = transfer function coefficients

‘zpk’ = zeros, poles, and overall gain

‘ss’ = state-space system representation

‘sos’ = transfer function coefficients of second-order sections

Functions, such as tf2zpk and zpk2ss, can convert between them.

The ba or tf format is a 2-tuple (b, a) representing a transfer function, where b is a length M+1 array of coefficients of the M-order numerator polynomial, and a is a length N+1 array of coefficients of the N-order denominator, as positive, descending powers of the transfer function variable. So the tuple of \(b = [b_0, b_1, ..., b_M]\) and \(a =[a_0, a_1, ..., a_N]\) can represent an analog filter of the form:

or a discrete-time filter of the form:

This “positive powers” form is found more commonly in controls engineering. If M and N are equal (which is true for all filters generated by the bilinear transform), then this happens to be equivalent to the “negative powers” discrete-time form preferred in DSP:

Although this is true for common filters, remember that this is not true in the general case. If M and N are not equal, the discrete-time transfer function coefficients must first be converted to the “positive powers” form before finding the poles and zeros.

This representation suffers from numerical error at higher orders, so other formats are preferred when possible.

The zpk format is a 3-tuple (z, p, k), where z is an M-length array of the complex zeros of the transfer function \(z = [z_0, z_1, ..., z_{M-1}]\), p is an N-length array of the complex poles of the transfer function \(p = [p_0, p_1, ..., p_{N-1}]\), and k is a scalar gain. These represent the digital transfer function:

or the analog transfer function:

Although the sets of roots are stored as ordered NumPy arrays, their ordering does not matter: ([-1, -2], [-3, -4], 1) is the same filter as ([-2, -1], [-4, -3], 1).

The ss format is a 4-tuple of arrays (A, B, C, D) representing the state-space of an N-order digital/discrete-time system of the form:

or a continuous/analog system of the form:

with P inputs, Q outputs and N state variables, where:

x is the state vector

y is the output vector of length Q

u is the input vector of length P

A is the state matrix, with shape (N, N)

B is the input matrix with shape (N, P)

C is the output matrix with shape (Q, N)

D is the feedthrough or feedforward matrix with shape (Q, P). (In cases where the system does not have a direct feedthrough, all values in D are zero.)

State-space is the most general representation and the only one that allows for multiple-input, multiple-output (MIMO) systems. There are multiple state-space representations for a given transfer function. Specifically, the “controllable canonical form” and “observable canonical form” have the same coefficients as the tf representation, and, therefore, suffer from the same numerical errors.

The sos format is a single 2-D array of shape (n_sections, 6), representing a sequence of second-order transfer functions which, when cascaded in series, realize a higher-order filter with minimal numerical error. Each row corresponds to a second-order tf representation, with the first three columns providing the numerator coefficients and the last three providing the denominator coefficients:

The coefficients are typically normalized, such that \(a_0\) is always 1. The section order is usually not important with floating-point computation; the filter output will be the same, regardless of the order.

The IIR filter design functions first generate a prototype analog low-pass filter with a normalized cutoff frequency of 1 rad/sec. This is then transformed into other frequencies and band types using the following substitutions:

\(s \rightarrow \frac{s}{\omega_0}\)

\(s \rightarrow \frac{\omega_0}{s}\)

\(s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}\)

\(s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}\)

Here, \(\omega_0\) is the new cutoff or center frequency, and \(\mathrm{BW}\) is the bandwidth. These preserve symmetry on a logarithmic frequency axis.

To convert the transformed analog filter into a digital filter, the bilinear transform is used, which makes the following substitution:

where T is the sampling time (the inverse of the sampling frequency).

The signal processing package provides many more filters as well.

A median filter is commonly applied when noise is markedly non-Gaussian or when it is desired to preserve edges. The median filter works by sorting all of the array pixel values in a rectangular region surrounding the point of interest. The sample median of this list of neighborhood pixel values is used as the value for the output array. The sample median is the middle-array value in a sorted list of neighborhood values. If there are an even number of elements in the neighborhood, then the average of the middle two values is used as the median. A general purpose median filter that works on N-D arrays is medfilt. A specialized version that works only for 2-D arrays is available as medfilt2d.

A median filter is a specific example of a more general class of filters called order filters. To compute the output at a particular pixel, all order filters use the array values in a region surrounding that pixel. These array values are sorted and then one of them is selected as the output value. For the median filter, the sample median of the list of array values is used as the output. A general-order filter allows the user to select which of the sorted values will be used as the output. So, for example, one could choose to pick the maximum in the list or the minimum. The order filter takes an additional argument besides the input array and the region mask that specifies which of the elements in the sorted list of neighbor array values should be used as the output. The command to perform an order filter is order_filter.

The Wiener filter is a simple deblurring filter for denoising images. This is not the Wiener filter commonly described in image-reconstruction problems but, instead, it is a simple, local-mean filter. Let \(x\) be the input signal, then the output is

where \(m_{x}\) is the local estimate of the mean and \(\sigma_{x}^{2}\) is the local estimate of the variance. The window for these estimates is an optional input parameter (default is \(3\times3\) ). The parameter \(\sigma^{2}\) is a threshold noise parameter. If \(\sigma\) is not given, then it is estimated as the average of the local variances.

The Hilbert transform constructs the complex-valued analytic signal from a real signal. For example, if \(x=\cos\omega n\), then \(y=\textrm{hilbert}\left(x\right)\) would return (except near the edges) \(y=\exp\left(j\omega n\right).\) In the frequency domain, the hilbert transform performs

where \(H\) is \(2\) for positive frequencies, \(0\) for negative frequencies, and \(1\) for zero-frequencies.

The functions iirdesign, iirfilter, and the filter design functions for specific filter types (e.g., ellip) all have a flag analog, which allows the design of analog filters as well.

The example below designs an analog (IIR) filter, obtains via tf2zpk the poles and zeros and plots them in the complex s-plane. The zeros at \(\omega \approx 150\) and \(\omega \approx 300\) can be clearly seen in the amplitude response.

Spectral analysis refers to investigating the Fourier transform [1] of a signal. Depending on the context, various names, like spectrum, spectral density or periodogram exist for the various spectral representations of the Fourier transform. [2] This section illustrates the most common representations by the example of a continuous-time sine wave signal of fixed duration. Then the use of the discrete Fourier transform [3] on a sampled version of that sine wave is discussed.

Separate subsections are devoted to the spectrum’s phase, estimating the power spectral density without (periodogram) and with averaging (welch) as well for non-equally spaced signals (lombscargle).

Note that the concept of Fourier series is closely related but differs in a crucial point: Fourier series have a spectrum made up of discrete-frequency harmonics, while in this section the spectra are continuous in frequency.

Consider a sine signal with amplitude \(a\), frequency \(f_x\) and duration \(\tau\), i.e.,

Since the \(\rect(t)\) function is one for \(|t|<1/2\) and zero for \(|t|>1/2\), it limits \(x(t)\) to the interval \([0, \tau]\). Expressing the sine by complex exponentials shows its two periodic components with frequencies \(\pm f_x\). We assume \(x(t)\) to be a voltage signal, so it has the unit \(\text{V}\).

In signal processing the integral of the absolute square \(|x(t)|^2\) is utilized to define energy and power of a signal, i.e.,

The power \(P_x\) can be interpreted as the energy \(E_x\) per unit time interval. Unit-wise, integrating over \(t\) results in multiplication with seconds. Hence, \(E_x\) has unit \(\text{V}^2\text{s}\) and \(P_x\) has the unit \(\text{V}^2\).

Applying the Fourier transform to \(x(t)\), i.e.,

results in two \(\sinc(f) := \sin(\pi f) /(\pi f)\) functions centered at \(\pm f_x\). The magnitude (absolute value) \(|X(f)|\) has two maxima located at \(\pm f_x\) with value \(|a|\tau/2\). It can be seen in the plot below that \(X(f)\) is not concentrated around the main lobes at \(\pm f_x\), but contains side lobes with heights decreasing proportional to \(1/(\tau f)\). This so-called “spectral leakage” [4] is caused by confining the sine to a finite interval. Note that the shorter the signal duration \(\tau\) is, the higher the leakage. To be independent of the signal duration, the so-called “amplitude spectrum” \(X(f)/\tau\) can be used instead of the spectrum \(X(f)\). Its value at \(f\) corresponds to the amplitude of the complex exponential \(\exp(\jj2\pi f t)\).

Due to Parseval’s theorem, the energy can be calculated from its Fourier transform \(X(f)\) by

as well. E.g., it can be shown by direct calculation that the energy of \(X(f)\) of Eq. (4) is \(|a|^2\tau/2\). Hence, the signal’s power in a frequency band \([f_a, f_b]\) can be determined with

Thus the function \(|X(f)|^2\) can be defined as the so-called “energy spectral density and \(S_{xx}(f) := |X(f)|^2 / \tau\) as “power spectral density” (PSD) of \(x(t)\). Instead of the PSD, the so-called “amplitude spectral density” \(X(f) / \sqrt{\tau}\) is also used, which still contains the phase information. Its absolute square is the PSD and thus it is closely related to the concept of the root-mean-square (RMS) value \(\sqrt{P_x}\) of a signal.

In summary, this subsection presented five ways to represent a spectrum:

Energy Spectral Density

Power Spectral Density (PSD)

Amplitude Spectral Density

\(X(f) / \sqrt{\tau}\)

Magnitude at \(\pm f_x\):

\(\frac{1}{2}|a|\tau\)

\(\frac{1}{4}|a|^2\tau^2\)

\(\frac{1}{4}|a|^2\tau\)

\(\frac{1}{2}|a|\sqrt{\tau}\)

\(\text{V} / \text{Hz}\)

\(\text{V}^2\text{s} / \text{Hz}\)

\(\text{V}^2 / \text{Hz}\)

\(\text{V} / \sqrt{\text{Hz}}\)

Note that the units presented in the table above are not unambiguous, e.g., \(\text{V}^2\text{s} / \text{Hz} = \text{V}^2\text{s}^2 = \text{V}^2/ \text{Hz}^2\). When using the absolute value of \(|X(f) / \tau|\) of the amplitude spectrum, it is called a magnitude spectrum. Furthermore, note that the naming scheme of the representations is not consistent and varies in literature.

For real-valued signals the so-called “onesided” spectral representation is often utilized. It only uses the non-negative frequencies (due to \(X(-f)= \conj{X}(f)\) if \(x(t)\in\IR\)). Sometimes the values of the negative frequencies are added to their positive counterparts. Then the amplitude spectrum allows to read off the full (not half) amplitude sine of \(x(t)\) at \(f_x\) and the area of an interval in the PSD represents its full (not half) power. Note that for amplitude spectral densities the positive values are not doubled but multiplied by \(\sqrt{2}\), since it is the square root of the PSD. Furthermore, there is no canonical way for naming a doubled spectrum.

The following plot shows three different spectral representations of four sine signals \(x(t)\) of Eq. (1) with different amplitudes \(a\) and durations \(\tau\). For less clutter, the spectra are centered at \(f_x\) and being are plotted next to each other:

Note that depending on the representation, the height of the peaks vary. Only the interpretation of the magnitude spectrum is straightforward: The peak at \(f_x\) in the second plot represents half the magnitude \(|a|\) of the sine signal. For all other representations the duration \(\tau\) needs to be taken into account to extract information about the signal’s amplitude.

In practice sampled signals are widely used. I.e., the signal is represented by \(n\) samples \(x_k := x(kT)\), \(k=0, \ldots, n-1\), where \(T\) is the sampling interval, \(\tau:=nT\) the signal’s duration and \(f_S := 1/T\) the sampling frequency. Note that the continuous signal needs to be band-limited to \([-f_S/2, f_S/2]\) to avoid aliasing, with \(f_S/2\) being called Nyquist frequency. [5] Replacing the integral by a sum to calculate the signal’s energy and power, i.e.,

delivers the identical result as in the continuous time case of Eq. (2). The discrete Fourier transform (DFT) and its inverse (as implemented using efficient FFT calculations in the scipy.fft module) is given by

The DFT and can be interpreted as an unscaled sampled version of the continuous Fourier transform of Eq. (3), i.e.,

The following plot shows the magnitude spectrum of two sine signals with unit amplitude and frequencies of 20 Hz and 20.5 Hz. The signal is made up of \(n=100\) samples with a sampling interval of \(T=10\) ms resulting in a duration of \(\tau=1\) s and a sampling frequency of \(f_S=100\) Hz.

The interpretation of the 20 Hz signal seems straightforward: All values are zero except at 20 Hz. There it is 0.5, which corresponds to half the amplitude of the input signal in accordance with Eq. (1). The peak of the 20.5 Hz signal on the other hand is dispersed along the frequency axis. Eq. (3) shows that this difference is caused by the reason that 20 Hz is a multiple of the bin width of 1 Hz whereas 20.5 Hz is not. The following plot illustrates this by overlaying continuous spectrum over the sampled one:

That a slight variation in frequency produces significantly different looking magnitude spectra is obviously undesirable behavior for many practical applications. The following two common techniques can be utilized to improve a spectrum:

The so-called “zero-padding” decreases \(\Delta f\) by appending zeros to the end of the signal. To oversample the frequency q times, pass the parameter n=q*n_x to the fft / rfft function with n_x being the length of the input signal.

The second technique is called windowing, i.e., multiplying the input signal with a suited function such that typically the secondary lobes are suppressed at the cost of widening the main lobe. The windowed DFT can be expressed as

where \(w_k\), \(k=0,\ldots,n-1\) is the sampled window function. To calculate the sampled versions of the spectral representations given in the previous subsection, the following normalization constants

need to be utilized. The first one ensures that a peak in the spectrum is consistent with the signal’s amplitude at that frequency. E.g., the magnitude spectrum can be expressed by \(|X^w_l / c^\text{amp}|\). The second constant guarantees that the power of a frequency interval as defined in Eq. (5) is consistent. The absolute values are needed since complex-valued windows are not forbidden.

The following plot shows the result of applying a hann window and three times over-sampling to \(x(t)\):

Now both lobes look almost identical and the side lobes are well suppressed. The maximum of the 20.5 Hz spectrum is also very close to the expected height of one half.

Spectral energy and spectral power can be calculated analogously to Eq. (4), yielding in identical results, i.e.,

This formulation is not to be confused with the special case of a rectangular window (or no window), i.e., \(w_k = 1\), \(X^w_l=X_l\), \(c^\text{den}=\sqrt{n}\), which results in

The windowed frequency-discrete power spectral density

is defined over the frequency range \([0, f_S)\) and can be interpreted as power per frequency interval \(\Delta f\). Integrating over a frequency band \([l_a\Delta f, l_b\Delta f)\), like in Eq. (5), becomes the sum

The windowed frequency-discrete energy spectral density \(\tau S^w_{xx}\) can be defined analogously.

The discussion above shows that sampled versions of the spectral representations as in the continuous-time case can be defined. The following tables summarizes these:

Energy Spectral Density

Power Spectral Density (PSD)

Amplitude Spectral Density

\(\tau X^w_l / c^\text{amp}\)

\(X^w_l / c^\text{amp}\)

\(\tau T |X^w_l / c^\text{den}|^2\)

\(T |X^w_l / c^\text{den}|^2\)

\(\sqrt{T} X^w_l / c^\text{den}\)

\(\text{V} / \text{Hz}\)

\(\text{V}^2\text{s} / \text{Hz}\)

\(\text{V}^2 / \text{Hz}\)

\(\text{V} / \sqrt{\text{Hz}}\)

Note that for the densities, the magnitude values at \(\pm f_x\) differ to the continuous time case due the change from integration to summation for determining spectral energy/power.

Though the hann window is the most common window function used in spectral analysis, other windows exist. The following plot shows the magnitude spectrum of various window functions of the windows submodule. It may be interpreted as the lobe shape of a single frequency input signal. Note that only the right half is shown and the \(y\)-axis is in decibel, i.e., it is logarithmically scaled.

This plot shows that the choice of window function is typically a trade-off between width of the main lobes and the height of the side lobes. Note that the boxcar window corresponds to a \(\rect\) function, i.e., to no windowing. Furthermore, many of the depicted windows are more frequently used in filter design than in spectral analysis.

The phase (i.e., angle) of the Fourier transform is typically utilized for investigating the time delay of the spectral components of a signal passing through a system like a filter. In the following example the standard test signal, an impulse with unit power, is passed through a simple filter, which delays the input by three samples. The input consists of \(n=50\) samples with sampling interval \(T = 1\) s. The plot shows magnitude and phase over frequency of the input and the output signal:

The input has a unit magnitude and zero-phase Fourier transform, which is the reason for the use as a test signal. The output has also unit magnitude but a linearly falling phase with a slope of \(-6\pi\). This is expected, since delaying a signal \(x(t)\) by \(\Delta t\) produces an additional linear phase term in its Fourier transform, i.e.,

Note that in the plot the phase is not limited to the interval \((+\pi, \pi]\) (output of angle) and hence does not have any discontinuities. This is achieved by utilizing the numpy.unwrap function. If the transfer function of the filter is known, freqz can be used to determine the spectral response of a filter directly.

The periodogram function calculates a power spectral density (scaling='density') or a squared magnitude spectrum (scaling='spectrum'). To obtain a smoothed periodogram, the welch function can be used. It does the smoothing by dividing the input signal into overlapping segments, to then calculate the windowed DFT of each segment. The result is to the average of those DFTs.

The example below shows the squared magnitude spectrum and the power spectral density of a signal made up of a \(1.27\,\text{kHz}\) sine signal with amplitude \(\sqrt{2}\,\text{V}\) and additive gaussian noise having a spectral power density with mean of \(10^{-3}\,\text{V}^2/\text{Hz}\).

The plots shows that the welch function produces a much smoother noise floor at the expense of the frequency resolution. Due to the smoothing the height of sine’s lobe is wider and not as high as in the periodogram. The left plot can be used to read the height of the lobe, i.e., half sine’s squared magnitude of \(1\,\text{V}^2\). The right plot can be used to determine the noise floor of \(10^{-3}\,\text{V}^2/\text{Hz}\). Note that the lobe height of the averaged squared magnitude spectrum is not exactly one due to limited frequency resolution. Either zero-padding (e.g., passing nfft=4*len(x) to welch) or reducing the number of segments by increasing the segment length (setting parameter nperseg) could be utilized to increase the number of frequency bins.

Least-squares spectral analysis (LSSA) [6] [7] is a method of estimating a frequency spectrum, based on a least-squares fit of sinusoids to data samples, similar to Fourier analysis. Fourier analysis, the most used spectral method in science, generally boosts long-periodic noise in long-gapped records; LSSA mitigates such problems.

The Lomb-Scargle method performs spectral analysis on unevenly-sampled data and is known to be a powerful way to find, and test the significance of, weak periodic signals.

For a time series comprising \(N_{t}\) measurements \(X_{j}\equiv X(t_{j})\) sampled at times \(t_{j}\), where \((j = 1, \ldots, N_{t})\), assumed to have been scaled and shifted, such that its mean is zero and its variance is unity, the normalized Lomb-Scargle periodogram at frequency \(f\) is

Here, \(\omega \equiv 2\pi f\) is the angular frequency. The frequency-dependent time offset \(\tau\) is given by

The lombscargle function calculates the periodogram using a slightly modified algorithm created by Zechmeister and Kürster [8], which allows for the weighting of individual samples and calculating an unknown offset (also called a “floating-mean”) for each frequency independently.

This section gives some background information on using the ShortTimeFFT class: The short-time Fourier transform (STFT) can be utilized to analyze the spectral properties of signals over time. It divides a signal into overlapping chunks by utilizing a sliding window and calculates the Fourier transform of each chunk. For a continuous-time complex-valued signal \(x(t)\) the STFT is defined [9] as

where \(w(t)\) is a complex-valued window function with its complex conjugate being \(\conj{w(t)}\). It can be interpreted as determining the scalar product of \(x\) with the window \(w\) which is translated by the time \(t\) and then modulated (i.e., frequency-shifted) by the frequency \(f\). For working with sampled signals \(x[k] := x(kT)\), \(k\in\IZ\), with sampling interval \(T\) (being the inverse of the sampling frequency fs), the discrete version, i.e., only evaluating the STFT at discrete grid points \(S[q, p] := S(q \Delta f, p\Delta t)\), \(q,p\in\IZ\), needs to be used. It can be formulated as

with p representing the time index of \(S\) with time interval \(\Delta t := h T\), \(h\in\IN\) (see delta_t), which can be expressed as the hop size of \(h\) samples. \(q\) represents the frequency index of \(S\) with step size \(\Delta f := 1 / (N T)\) (see delta_f), which makes it FFT compatible. \(w[m] := w(mT)\), \(m\in\IZ\) is the sampled window function.

To be more aligned to the implementation of ShortTimeFFT, it makes sense to reformulate Eq. (8) as a two-step process:

Extract the \(p\)-th slice by windowing with the window \(w[m]\) made up of \(M\) samples (see m_num) centered at \(t[p] := p \Delta t = h T\) (see delta_t), i.e.,

where the integer \(\lfloor M/2\rfloor\) represents M//2, i.e., it is the mid point of the window (m_num_mid). For notational convenience, \(x[k]:=0\) for \(k\not\in\{0, 1, \ldots, N-1\}\) is assumed. In the subsection Sliding Windows the indexing of the slices is discussed in more detail.

Then perform a discrete Fourier transform (i.e., an FFT) of \(x_p[m]\).

Note that a linear phase \(\phi_m\) (see phase_shift) can be specified, which corresponds to shifting the input by \(\phi_m\) samples. The default is \(\phi_m = \lfloor M/2\rfloor\) (corresponds per definition to phase_shift = 0), which suppresses linear phase components for unshifted signals. Furthermore, the FFT may be oversampled by padding \(w[m]\) with zeros. This can be achieved by specifying mfft to be larger than the window length m_num—this sets \(M\) to mfft (implying that also \(w[m]:=0\) for \(m\not\in\{0, 1, \ldots, M-1\}\) holds).

The inverse short-time Fourier transform (istft) is implemented by reversing these two steps:

Perform the inverse discrete Fourier transform, i.e.,

Sum the shifted slices weighted by \(w_d[m]\) to reconstruct the original signal, i.e.,

for \(k \in [0, \ldots, n-1]\). \(w_d[m]\) is the so-called dual window of \(w[m]\) and is also made up of \(M\) samples.

Note that an inverse STFT does not necessarily exist for all windows and hop sizes. For a given window \(w[m]\) the hop size \(h\) must be small enough to ensure that every sample of \(x[k]\) is touched by a non-zero value of at least one window slice. This is sometimes referred as the “non-zero overlap condition” (see check_NOLA). Some more details are given in the subsection Inverse STFT and Dual Windows.

This subsection discusses how the sliding window is indexed in the ShortTimeFFT by means of an example: Consider a window of length 6 with a hop interval of two and a sampling interval T of one, e.g., ShortTimeFFT (np.ones(6), 2, fs=1). The following image schematically depicts the first four window positions also named time slices:

The x-axis denotes the time \(t\), which corresponds to the sample index k indicated by the bottom row of blue boxes. The y-axis denotes the time slice index \(p\). The signal \(x[k]\) starts at index \(k=0\) and is marked by a light blue background. Per definition the zeroth slice (\(p=0\)) is centered at \(t=0\). The center of each slice (m_num_mid), here being the sample 6//2=3, is marked by the text “mid”. By default the stft calculates all slices which have some overlap with the signal. Hence the first slice is at p_min = -1 with the lowest sample index being k_min = -5. The first sample index unaffected by a slice not sticking out to the left of the signal is \(p_{lb} = 2\) and the first sample index unaffected by border effects is \(k_{lb} = 5\). The property lower_border_end returns the tuple \((k_{lb}, p_{lb})\).

The behavior at the end of the signal is depicted for a signal with \(n=50\) samples below, as indicated by the blue background:

Here the last slice has index \(p=26\). Hence, following Python convention of the end index being outside the range, p_max = 27 indicates the first slice not touching the signal. The corresponding sample index is k_max = 55. The first slice, which sticks out to the right is \(p_{ub} = 24\) with its first sample at \(k_{ub}=45\). The function upper_border_begin returns the tuple \((k_{ub}, p_{ub})\).

The term dual window stems from frame theory [10] where a frame is a series expansion which can represent any function in a given Hilbert space. There the expansions \(\{g_k\}\) and \(\{h_k\}\) are dual frames if for all functions \(f\) in the given Hilbert space \(\mathcal{H}\)

holds, where \(\langle ., .\rangle\) denotes the scalar product of \(\mathcal{H}\). All frames have dual frames [10].

An STFT evaluated only at discrete grid points \(S(q \Delta f, p\Delta t)\) is called a “Gabor frame” in literature [9] [10]. Since the support of the window \(w[m]\) is limited to a finite interval, the ShortTimeFFT falls into the class of the so-called “painless non-orthogonal expansions” [9]. In this case the dual windows always have the same support and can be calculated by means of inverting a diagonal matrix. A rough derivation only requiring some understanding of manipulating matrices will be sketched out in the following:

Since the STFT given in Eq. (8) is a linear mapping in \(x[k]\), it can be expressed in vector-matrix notation. This allows us to express the inverse via the formal solution of the linear least squares method (as in lstsq), which leads to a beautiful and simple result.

We begin by reformulating the windowing of Eq. (9)

where the \(M\times N\) matrix \(\vb{W}_{\!p}\) has only non-zeros entries on the \((ph)\)-th minor diagonal, i.e.,

with \(\delta_{k,l}\) being the Kronecker Delta. Eq. (10) can be expressed as

which allows the STFT of the \(p\)-th slice to be written as

Due to the scaling factor of \(M^{-1/2}\), \(\vb{F}\) is unitary, i.e., the inverse equals its conjugate transpose meaning \(\conjT{\vb{F}}\vb{F} = \vb{I}\). Other scalings, e.g., like in Eq. (6), are allowed as well, but would in this section make the notation slightly more complicated.

To obtain a single vector-matrix equation for the STFT, the slices are stacked into one vector, i.e.,

where \(P\) is the number of columns of the resulting STFT. To invert this equation the Moore-Penrose inverse \(\vb{G}^\dagger\) can be utilized

is invertible. Then \(\vb{x} = \vb{G}^\dagger\vb{G}\,\vb{x} = \inv{\conjT{\vb{G}}\vb{G}}\,\conjT{\vb{G}}\vb{G}\,\vb{x}\) obviously holds. \(\vb{D}\) is always a diagonal matrix with non-negative diagonal entries. This becomes clear, when simplifying \(\vb{D}\) further to

due to \(\vb{F}\) being unitary. Furthermore

shows that \(\vb{D}_p\) is a diagonal matrix with non-negative entries. Hence, summing \(\vb{D}_p\) preserves that property. This allows to simplify Eq. (15) further, i.e.,

Utilizing Eq. (12), (17), (18), \(\vb{U}_p=\vb{W}_{\!p}\vb{D}^{-1}\) can be expressed as

This shows \(\vb{U}_p\) has the identical structure as \(\vb{W}_p\) in Eq. (12), i.e., having only non-zero entries on the \((ph)\)-th minor diagonal. The sum term in the inverse can be interpreted as sliding \(|w[\mu]|^2\) over \(w[m]\) (with an incorporated inversion), so only components overlapping with \(w[m]\) have an effect. Hence, all \(U_p[m, k]\) far enough from the border are identical windows. To circumvent border effects, \(x[k]\) is padded with zeros, enlarging \(\vb{U}\) so all slices which touch \(x[k]\) contain the identical dual window

Since \(w[m] = 0\) holds for \(m \not\in\{0, \ldots, M-1\}\), it is only required to sum over the indexes \(\eta\) fulfilling \(|\eta| < M/h\). The second expression is an alternate form by summing from index \(l=0\) to \(M\) in index increments of \(h\). \(\delta_{l+m,\eta h}\) is Kronecker delta notation for (l+m) % h == 0. The name dual window can be justified by inserting Eq. (14) into Eq. (19), i.e.,

showing that \(\vb{U}_p\) and \(\vb{W}_{\!p}\) are interchangeable. Due \(\vb{U}_p\) and \(\vb{W}_{\!p}\) having the same structure given in Eq. (11), Eq. (22) contains a general condition for all possible dual windows, i.e.,

which can be reformulated into

where \(\vb{1}\in\IR^h\) is a vector of ones. The reason that \(\vb{V}\) has only \(h\) columns is that the \(i\)-th and \((i+m)\)-th row, \(i\in\IN\), in Eq. (22) are identical. Hence there are only \(h\) distinct equations.

Of practical interest is finding the valid dual window \(\vb{u}_d\) closest to a given vector \(\vb{d}\in\IC^M\). By utilizing an \(h\)-dimensional vector \(\vb{\lambda}\) of Lagrange multipliers, we obtain the convex quadratic programming problem

A closed form solution can be calculated by inverting the \(2\times2\) block matrix symbolically, which gives

with \(\eta,\xi,\zeta\in\IZ\). Note that the first term \(\vb{w}_d\) is equal to the solution given in Eq. (21) and that the inverse of \(\conjT{\vb{V}}\vb{V}\) must exist or else the STFT is not invertible. When \(\vb{d}=\vb{0}\), the solution \(\vb{w}_d\) is obtained. Hence, \(\vb{w}_d\) minimizes the \(L^2\)-norm \(\lVert\vb{u}\rVert\), which is the justification for its name “canonical dual window”. Sometimes it is more desirable to find the closest vector in regard to direction and ignoring the vector length. This can be achieved by introducing a scaling factor \(\alpha\in\IC\) to minimize \(\lVert\alpha\vb{d} - \vb{u}\rVert^2\). Since Eq. (25) already provides a general solution, we can write

The case where the window \(w[m]\) and the dual window \(u[m]\) are equal can be easily derived from Eq. (23) resulting in \(h\) conditions of the form

Note that each window sample \(w[m]\) appears only once in the \(h\) equations. To find a closest window \(\vb{w}\) for given window \(\vb{d}\) is straightforward: Partition \(\vb{d}\) according to Eq. (26) and normalize the length of each partition to unity. In this case \(w[m]\) is also a canonical dual window, which can be seen by recognizing that setting \(u[m]=w[m]\) in Eq. (24) is equivalent of the denominator in Eq. (21) being unity.

Furthermore, if Eq. (26) holds, the matrix \(\vb{D}\) of Eq. (16) is the identity matrix making the STFT \(\vb{G}\) a unitary mapping, i.e., \(\conjT{\vb{G}}\vb{G}=\vb{I}\). Note that this holds only when a unitary DFT of Eq. (13) is utilized. The ShortTimeFFT implementation uses the standard DFT of Eq. (6). Hence, there the scalar product in the STFT space needs to be scaled by \(1/M\) to ensure that the key property of unitary mappings, the equality of the scalar products, holds. I.e.,

with \(S_{x,y}\) being the STFT of \(x,y\). Alternatively, the window can be scaled by \(1/\sqrt{M}\) and the dual by \(\sqrt{M}\) to obtain a unitary mapping, which is implemented in from_win_equals_dual.

The functions stft, istft, and the spectrogram predate the ShortTimeFFT implementation. This section discusses the key differences between the older “legacy” and the newer ShortTimeFFT implementations. The main motivation for a rewrite was the insight that integrating dual windows could not be done in a sane way without breaking compatibility. This opened the opportunity for rethinking the code structure and the parametrization, thus making some implicit behavior more explicit.

The following example compares the two STFTs of a complex valued chirp signal with a negative slope:

That the ShortTimeFFT produces 3 more time slices than the legacy version is the main difference. As laid out in the Sliding Windows section, all slices which touch the signal are incorporated in the new version. This has the advantage that the STFT can be sliced and reassembled as shown in the ShortTimeFFT code example. Furthermore, using all touching slices makes the ISTFT more robust in the case of windows that are zero somewhere.

Note that the slices with identical time stamps produce equal results (up to numerical accuracy), i.e.:

Generally, those additional slices contain non-zero values. Due to the large overlap in our example, they are quite small. E.g.:

The ISTFT can be utilized to reconstruct the original signal:

Note that the legacy implementation returns a signal which is longer than the original. On the other hand, the new istft allows to explicitly specify the start index k0 and the end index k1 of the reconstructed signal. The length discrepancy in the old implementation is caused by the fact that the signal length is not a multiple of the slices.

Further differences between the new and legacy versions in this example are:

The parameter fft_mode='centered' ensures that the zero frequency is vertically centered for two-sided FFTs in the plot. With the legacy implementation, fftshift needs to be utilized. fft_mode='twosided' produces the same behavior as the old version.

The parameter phase_shift=None ensures identical phases of the two versions. ShortTimeFFT’s default value of 0 produces STFT slices with an additional linear phase term.

A spectrogram is defined as the absolute square of the STFT [9]. The spectrogram provided by the ShortTimeFFT sticks to that definition, i.e.:

On the other hand, the legacy spectrogram provides another STFT implementation with the key difference being the different handling of the signal borders. The following example shows how to use the ShortTimeFFT to obtain an identical SFT as produced with the legacy spectrogram:

The difference from the other STFTs is that the time slices do not start at 0 but at nperseg//2, i.e.:

Furthermore, only slices which do not stick out to the right are returned, centering the last slice at 4.875 s, which makes it shorter than with the default stft parametrization.

Using the mode parameter, the legacy spectrogram can also return the ‘angle’, ‘phase’, ‘psd’ or the ‘magnitude’. The scaling behavior of the legacy spectrogram is not straightforward, since it depends on the parameters mode, scaling and return_onesided. There is no direct correspondence for all combinations in the ShortTimeFFT, since it provides only ‘magnitude’, ‘psd’ or no scaling of the window at all. The following table shows those correspondences:

When using onesided output on complex-valued input signals, the old spectrogram switches to two-sided mode. The ShortTimeFFT raises a TypeError, since the utilized rfft function only accepts real-valued inputs. Consult the Spectral Analysis section above for a discussion on the various spectral representations which are induced by the various parameterizations.

Some further reading and related software:

“Fourier transform”, Wikipedia, https://en.wikipedia.org/wiki/Fourier_transform

“Spectral density”, Wikipedia, https://en.wikipedia.org/wiki/Spectral_density

“Discrete Fourier transform”, Wikipedia, https://en.wikipedia.org/wiki/Discrete_Fourier_transform

“Spectral Leakage”, Wikipedia, https://en.wikipedia.org/wiki/Spectral_leakage

“Nyquist–Shannon sampling theorem”, Wikipedia, https://en.wikipedia.org/wiki/Nyquist-Shannon_sampling_theorem

N.R. Lomb “Least-squares frequency analysis of unequally spaced data”, Astrophysics and Space Science, vol 39, pp. 447-462, 1976

J.D. Scargle “Studies in astronomical time series analysis. II - Statistical aspects of spectral analysis of unevenly spaced data”, The Astrophysical Journal, vol 263, pp. 835-853, 1982

M. Zechmeister and M. Kürster, “The generalised Lomb-Scargle periodogram. A new formalism for the floating-mean and Keplerian periodograms,” Astronomy and Astrophysics, vol. 496, pp. 577-584, 2009

Karlheinz Gröchenig: “Foundations of Time-Frequency Analysis”, Birkhäuser Boston 2001, DOI:10.1007/978-1-4612-0003-1

Ole Christensen: “An Introduction to Frames and Riesz Bases”, Birkhäuser Boston 2016, DOI:10.1007/978-3-319-25613-9

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import signal, datasets
>>> import matplotlib.pyplot as plt
```

Example 2 (unknown):
```unknown
>>> image = datasets.face(gray=True).astype(np.float32)
>>> derfilt = np.array([1.0, -2, 1.0], dtype=np.float32)
>>> ck = signal.cspline2d(image, 8.0)
>>> deriv = (signal.sepfir2d(ck, derfilt, [1]) +
...          signal.sepfir2d(ck, [1], derfilt))
```

Example 3 (unknown):
```unknown
laplacian = np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.float32)
deriv2 = signal.convolve2d(ck,laplacian,mode='same',boundary='symm')
```

Example 4 (unknown):
```unknown
>>> plt.figure()
>>> plt.imshow(image)
>>> plt.gray()
>>> plt.title('Original image')
>>> plt.show()
```

---

## lstsq#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html

**Contents:**
- lstsq#

Compute least-squares solution to the equation a @ x = b.

Compute a vector x such that the 2-norm |b - A x| is minimized.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Right hand side array

Cutoff for ‘small’ singular values; used to determine effective rank of a. Singular values smaller than cond * largest_singular_value are considered zero.

Discard data in a (may enhance performance). Default is False.

Discard data in b (may enhance performance). Default is False.

Whether to check that the input matrices contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Which LAPACK driver is used to solve the least-squares problem. Options are 'gelsd', 'gelsy', 'gelss'. Default ('gelsd') is a good choice. However, 'gelsy' can be slightly faster on many problems. 'gelss' was used historically. It is generally slow but uses less memory.

Added in version 0.17.0.

Least-squares solution.

Square of the 2-norm for each column in b - a x, if M > N and rank(A) == n (returns a scalar if b is 1-D). Otherwise a (0,)-shaped array is returned.

Singular values of a. The condition number of a is s[0] / s[-1].

If computation does not converge.

When parameters are not compatible.

linear least squares with non-negativity constraint

When 'gelsy' is used as a driver, residues is set to a (0,)-shaped array and s is always None.

Suppose we have the following data:

We want to fit a quadratic polynomial of the form y = a + b*x**2 to this data. We first form the “design matrix” M, with a constant column of 1s and a column containing x**2:

We want to find the least-squares solution to M.dot(p) = y, where p is a vector with length 2 that holds the parameters a and b.

Plot the data and the fitted curve.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import lstsq
>>> import matplotlib.pyplot as plt
```

Example 2 (unknown):
```unknown
>>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
>>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])
```

Example 3 (json):
```json
>>> M = x[:, np.newaxis]**[0, 2]
>>> M
array([[  1.  ,   1.  ],
       [  1.  ,   6.25],
       [  1.  ,  12.25],
       [  1.  ,  16.  ],
       [  1.  ,  25.  ],
       [  1.  ,  49.  ],
       [  1.  ,  72.25]])
```

Example 4 (unknown):
```unknown
>>> p, res, rnk, s = lstsq(M, y)
>>> p
array([ 0.20925829,  0.12013861])
```

---

## Distance computations (scipy.spatial.distance)#

**URL:** https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

**Contents:**
- Distance computations (scipy.spatial.distance)#
- Function reference#

Distance matrix computation from a collection of raw observation vectors stored in a rectangular array.

pdist(X[, metric, out])

Pairwise distances between observations in n-dimensional space.

cdist(XA, XB[, metric, out])

Compute distance between each pair of the two collections of inputs.

squareform(X[, force, checks])

Convert a vector-form distance vector to a square-form distance matrix, and vice-versa.

directed_hausdorff(u, v[, rng, seed])

Compute the directed Hausdorff distance between two 2-D arrays.

Predicates for checking the validity of distance matrices, both condensed and redundant. Also contained in this module are functions for computing the number of observations in a distance matrix.

is_valid_dm(D[, tol, throw, name, warning])

Return True if input array is a valid distance matrix.

is_valid_y(y[, warning, throw, name])

Return True if the input array is a valid condensed distance matrix.

Return the number of original observations that correspond to a square, redundant distance matrix.

Return the number of original observations that correspond to a condensed distance matrix.

Distance functions between two numeric vectors u and v. Computing distances over a large collection of vectors is inefficient for these functions. Use pdist for this purpose.

braycurtis(u, v[, w])

Compute the Bray-Curtis distance between two 1-D arrays.

Compute the Canberra distance between two 1-D arrays.

Compute the Chebyshev distance.

Compute the City Block (Manhattan) distance.

correlation(u, v[, w, centered])

Compute the correlation distance between two 1-D arrays.

Compute the Cosine distance between 1-D arrays.

Computes the Euclidean distance between two 1-D arrays.

jensenshannon(p, q[, base, axis, keepdims])

Compute the Jensen-Shannon distance (metric) between two probability arrays.

mahalanobis(u, v, VI)

Compute the Mahalanobis distance between two 1-D arrays.

minkowski(u, v[, p, w])

Compute the Minkowski distance between two 1-D arrays.

Return the standardized Euclidean distance between two 1-D arrays.

sqeuclidean(u, v[, w])

Compute the squared Euclidean distance between two 1-D arrays.

Distance functions between two boolean vectors (representing sets) u and v. As in the case of numerical vectors, pdist is more efficient for computing the distances between all pairs.

Compute the Dice dissimilarity between two boolean 1-D arrays.

Compute the Hamming distance between two 1-D arrays.

Compute the Jaccard dissimilarity between two boolean vectors.

kulczynski1(u, v, *[, w])

Compute the Kulczynski 1 dissimilarity between two boolean 1-D arrays.

rogerstanimoto(u, v[, w])

Compute the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.

russellrao(u, v[, w])

Compute the Russell-Rao dissimilarity between two boolean 1-D arrays.

sokalmichener(u, v[, w])

Compute the Sokal-Michener dissimilarity between two boolean 1-D arrays.

sokalsneath(u, v[, w])

Compute the Sokal-Sneath dissimilarity between two boolean 1-D arrays.

Compute the Yule dissimilarity between two boolean 1-D arrays.

hamming also operates over discrete numerical vectors.

---

## read#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html

**Contents:**
- read#

Return the sample rate (in samples/sec) and data from an LPCM WAV file.

Whether to read data as memory-mapped (default: False). Not compatible with some bit depths; see Notes. Only to be used on real files.

Added in version 0.12.0.

Sample rate of WAV file.

Data read from WAV file. Data-type is determined from the file; see Notes. Data is 1-D for 1-channel WAV, or 2-D of shape (Nsamples, Nchannels) otherwise. If a file-like input without a C-like file descriptor (e.g., io.BytesIO) is passed, this will not be writeable.

Common data types: [1]

32-bit floating-point

WAV files can specify arbitrary bit depth, and this function supports reading any integer PCM depth from 1 to 64 bits. Data is returned in the smallest compatible numpy int type, in left-justified format. 8-bit and lower is unsigned, while 9-bit and higher is signed.

For example, 24-bit data will be stored as int32, with the MSB of the 24-bit data stored at the MSB of the int32, and typically the least significant byte is 0x00. (However, if a file actually contains data past its specified bit depth, those bits will be read and output, too. [2])

This bit justification and sign matches WAV’s native internal format, which allows memory mapping of WAV files that use 1, 2, 4, or 8 bytes per sample (so 24-bit files cannot be memory-mapped, but 32-bit can).

IEEE float PCM in 32- or 64-bit format is supported, with or without mmap. Values exceeding [-1, +1] are not clipped.

Non-linear PCM (mu-law, A-law) is not supported.

IBM Corporation and Microsoft Corporation, “Multimedia Programming Interface and Data Specifications 1.0”, section “Data Format of the Samples”, August 1991 http://www.tactilemedia.com/info/MCI_Control_Info.html

Adobe Systems Incorporated, “Adobe Audition 3 User Guide”, section “Audio file formats: 24-bit Packed Int (type 1, 20-bit)”, 2007

Get the filename for an example .wav file from the tests/data directory.

Load the .wav file contents.

**Examples:**

Example 1 (sql):
```sql
>>> from os.path import dirname, join as pjoin
>>> from scipy.io import wavfile
>>> import scipy.io
```

Example 2 (unknown):
```unknown
>>> data_dir = pjoin(dirname(scipy.io.__file__), 'tests', 'data')
>>> wav_fname = pjoin(data_dir, 'test-44100Hz-2ch-32bit-float-be.wav')
```

Example 3 (unknown):
```unknown
>>> samplerate, data = wavfile.read(wav_fname)
>>> print(f"number of channels = {data.shape[1]}")
number of channels = 2
>>> length = data.shape[0] / samplerate
>>> print(f"length = {length}s")
length = 0.01s
```

Example 4 (typescript):
```typescript
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> time = np.linspace(0., length, data.shape[0])
>>> plt.plot(time, data[:, 0], label="Left channel")
>>> plt.plot(time, data[:, 1], label="Right channel")
>>> plt.legend()
>>> plt.xlabel("Time [s]")
>>> plt.ylabel("Amplitude")
>>> plt.show()
```

---

## loadarff#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html

**Contents:**
- loadarff#

The data is returned as a record array, which can be accessed much like a dictionary of NumPy arrays. For example, if one of the attributes is called ‘pressure’, then its first 10 data points can be accessed from the data record array like so: data['pressure'][0:10]

File-like object to read from, or filename to open.

The data of the arff file, accessible by attribute names.

Contains information about the arff file such as name and type of attributes, the relation (name of the dataset), etc.

This is raised if the given file is not ARFF-formatted.

The ARFF file has an attribute which is not supported yet.

This function should be able to read most arff files. Not implemented functionality include:

string type attributes

It can read files with numeric and nominal attributes. It cannot read files with sparse data ({} in the file). However, this function can read files with missing data (? in the file), representing the data points as NaNs.

**Examples:**

Example 1 (python):
```python
>>> from scipy.io import arff
>>> from io import StringIO
>>> content = """
... @relation foo
... @attribute width  numeric
... @attribute height numeric
... @attribute color  {red,green,blue,yellow,black}
... @data
... 5.0,3.25,blue
... 4.5,3.75,green
... 3.0,4.00,red
... """
>>> f = StringIO(content)
>>> data, meta = arff.loadarff(f)
>>> data
array([(5.0, 3.25, 'blue'), (4.5, 3.75, 'green'), (3.0, 4.0, 'red')],
      dtype=[('width', '<f8'), ('height', '<f8'), ('color', '|S6')])
>>> meta
Dataset: foo
    width's type is numeric
    height's type is numeric
    color's type is nominal, range is ('red', 'green', 'blue', 'yellow', 'black')
```

---

## invpascal#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.invpascal.html

**Contents:**
- invpascal#

Returns the inverse of the n x n Pascal matrix.

The Pascal matrix is a matrix containing the binomial coefficients as its elements.

The size of the matrix to create; that is, the result is an n x n matrix.

Must be one of ‘symmetric’, ‘lower’, or ‘upper’. Default is ‘symmetric’.

If exact is True, the result is either an array of type numpy.int64 (if n <= 35) or an object array of Python integers. If exact is False, the coefficients in the matrix are computed using scipy.special.comb with exact=False. The result will be a floating point array, and for large n, the values in the array will not be the exact coefficients.

The inverse of the Pascal matrix.

Added in version 0.16.0.

“Pascal matrix”, https://en.wikipedia.org/wiki/Pascal_matrix

Cohen, A. M., “The inverse of a Pascal matrix”, Mathematical Gazette, 59(408), pp. 111-112, 1975.

An example of the use of kind and exact:

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import invpascal, pascal
>>> invp = invpascal(5)
>>> invp
array([[  5, -10,  10,  -5,   1],
       [-10,  30, -35,  19,  -4],
       [ 10, -35,  46, -27,   6],
       [ -5,  19, -27,  17,  -4],
       [  1,  -4,   6,  -4,   1]])
```

Example 2 (json):
```json
>>> p = pascal(5)
>>> p.dot(invp)
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  1.]])
```

Example 3 (json):
```json
>>> invpascal(5, kind='lower', exact=False)
array([[ 1., -0.,  0., -0.,  0.],
       [-1.,  1., -0.,  0., -0.],
       [ 1., -2.,  1., -0.,  0.],
       [-1.,  3., -3.,  1., -0.],
       [ 1., -4.,  6., -4.,  1.]])
```

---

## sinm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sinm.html

**Contents:**
- sinm#

Compute the matrix sine.

This routine uses expm to compute the matrix exponentials.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Euler’s identity (exp(i*theta) = cos(theta) + i*sin(theta)) applied to a matrix:

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import expm, sinm, cosm
```

Example 2 (json):
```json
>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
```

---

## quad#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html

**Contents:**
- quad#

Compute a definite integral.

Integrate func from a to b (possibly infinite interval) using a technique from the Fortran library QUADPACK.

A Python function or method to integrate. If func takes many arguments, it is integrated along the axis corresponding to the first argument.

If the user desires improved integration performance, then f may be a scipy.LowLevelCallable with one of the signatures:

The user_data is the data contained in the scipy.LowLevelCallable. In the call forms with xx, n is the length of the xx array which contains xx[0] == x and the rest of the items are numbers contained in the args argument of quad.

In addition, certain ctypes call signatures are supported for backward compatibility, but those should not be used in new code.

Lower limit of integration (use -numpy.inf for -infinity).

Upper limit of integration (use numpy.inf for +infinity).

Extra arguments to pass to func.

Non-zero to return a dictionary of integration information. If non-zero, warning messages are also suppressed and the message is appended to the output tuple.

Indicate if the function’s (func) return type is real (complex_func=False: default) or complex (complex_func=True). In both cases, the function’s argument is real. If full_output is also non-zero, the infodict, message, and explain for the real and complex components are returned in a dictionary with keys “real output” and “imag output”.

The integral of func from a to b.

An estimate of the absolute error in the result.

A dictionary containing additional information.

A convergence message.

Appended only with ‘cos’ or ‘sin’ weighting and infinite integration limits, it contains an explanation of the codes in infodict[‘ierlst’]

Absolute error tolerance. Default is 1.49e-8. quad tries to obtain an accuracy of abs(i-result) <= max(epsabs, epsrel*abs(i)) where i = integral of func from a to b, and result is the numerical approximation. See epsrel below.

Relative error tolerance. Default is 1.49e-8. If epsabs <= 0, epsrel must be greater than both 5e-29 and 50 * (machine epsilon). See epsabs above.

An upper bound on the number of subintervals used in the adaptive algorithm.

A sequence of break points in the bounded integration interval where local difficulties of the integrand may occur (e.g., singularities, discontinuities). The sequence does not have to be sorted. Note that this option cannot be used in conjunction with weight.

String indicating weighting function. Full explanation for this and the remaining arguments can be found below.

Variables for use with weighting functions.

Optional input for reusing Chebyshev moments.

An upper bound on the number of Chebyshev moments.

Upper bound on the number of cycles (>=3) for use with a sinusoidal weighting and an infinite end-point.

n-dimensional integrals (uses quad recursively)

fixed-order Gaussian quadrature

integrator for sampled data

integrator for sampled data

for coefficients and roots of orthogonal polynomials

For valid results, the integral must converge; behavior for divergent integrals is not guaranteed.

Extra information for quad() inputs and outputs

If full_output is non-zero, then the third output argument (infodict) is a dictionary with entries as tabulated below. For infinite limits, the range is transformed to (0,1) and the optional outputs are given with respect to this transformed range. Let M be the input argument limit and let K be infodict[‘last’]. The entries are:

The number of function evaluations.

The number, K, of subintervals produced in the subdivision process.

A rank-1 array of length M, the first K elements of which are the left end points of the subintervals in the partition of the integration range.

A rank-1 array of length M, the first K elements of which are the right end points of the subintervals.

A rank-1 array of length M, the first K elements of which are the integral approximations on the subintervals.

A rank-1 array of length M, the first K elements of which are the moduli of the absolute error estimates on the subintervals.

A rank-1 integer array of length M, the first L elements of which are pointers to the error estimates over the subintervals with L=K if K<=M/2+2 or L=M+1-K otherwise. Let I be the sequence infodict['iord'] and let E be the sequence infodict['elist']. Then E[I[1]], ..., E[I[L]] forms a decreasing sequence.

If the input argument points is provided (i.e., it is not None), the following additional outputs are placed in the output dictionary. Assume the points sequence is of length P.

A rank-1 array of length P+2 containing the integration limits and the break points of the intervals in ascending order. This is an array giving the subintervals over which integration will occur.

A rank-1 integer array of length M (=limit), containing the subdivision levels of the subintervals, i.e., if (aa,bb) is a subinterval of (pts[1], pts[2]) where pts[0] and pts[2] are adjacent elements of infodict['pts'], then (aa,bb) has level l if |bb-aa| = |pts[2]-pts[1]| * 2**(-l).

A rank-1 integer array of length P+2. After the first integration over the intervals (pts[1], pts[2]), the error estimates over some of the intervals may have been increased artificially in order to put their subdivision forward. This array has ones in slots corresponding to the subintervals for which this happens.

Weighting the integrand

The input variables, weight and wvar, are used to weight the integrand by a select list of functions. Different integration methods are used to compute the integral with these weighting functions, and these do not support specifying break points. The possible values of weight and the corresponding weighting functions are.

g(x) = ((x-a)**alpha)*((b-x)**beta)

g(x)*log(x-a)*log(b-x)

wvar holds the parameter w, (alpha, beta), or c depending on the weight selected. In these expressions, a and b are the integration limits.

For the ‘cos’ and ‘sin’ weighting, additional inputs and outputs are available.

For weighted integrals with finite integration limits, the integration is performed using a Clenshaw-Curtis method, which uses Chebyshev moments. For repeated calculations, these moments are saved in the output dictionary:

The maximum level of Chebyshev moments that have been computed, i.e., if M_c is infodict['momcom'] then the moments have been computed for intervals of length |b-a| * 2**(-l), l=0,1,...,M_c.

A rank-1 integer array of length M(=limit), containing the subdivision levels of the subintervals, i.e., an element of this array is equal to l if the corresponding subinterval is |b-a|* 2**(-l).

A rank-2 array of shape (25, maxp1) containing the computed Chebyshev moments. These can be passed on to an integration over the same interval by passing this array as the second element of the sequence wopts and passing infodict[‘momcom’] as the first element.

If one of the integration limits is infinite, then a Fourier integral is computed (assuming w neq 0). If full_output is 1 and a numerical error is encountered, besides the error message attached to the output tuple, a dictionary is also appended to the output tuple which translates the error codes in the array info['ierlst'] to English messages. The output information dictionary contains the following entries instead of ‘last’, ‘alist’, ‘blist’, ‘rlist’, and ‘elist’:

The number of subintervals needed for the integration (call it K_f).

A rank-1 array of length M_f=limlst, whose first K_f elements contain the integral contribution over the interval (a+(k-1)c, a+kc) where c = (2*floor(|w|) + 1) * pi / |w| and k=1,2,...,K_f.

A rank-1 array of length M_f containing the error estimate corresponding to the interval in the same position in infodict['rslist'].

A rank-1 integer array of length M_f containing an error flag corresponding to the interval in the same position in infodict['rslist']. See the explanation dictionary (last entry in the output tuple) for the meaning of the codes.

Details of QUADPACK level routines

quad calls routines from the FORTRAN library QUADPACK. This section provides details on the conditions for each routine to be called and a short description of each routine. The routine called depends on weight, points and the integration limits a and b.

The following provides a short description from [1] for each routine.

is an integrator based on globally adaptive interval subdivision in connection with extrapolation, which will eliminate the effects of integrand singularities of several types. The integration is performed using a 21-point Gauss-Kronrod quadrature within each subinterval.

handles integration over infinite intervals. The infinite range is mapped onto a finite interval and subsequently the same strategy as in QAGS is applied.

serves the same purposes as QAGS, but also allows the user to provide explicit information about the location and type of trouble-spots i.e. the abscissae of internal singularities, discontinuities and other difficulties of the integrand function.

is an integrator for the evaluation of \(\int^b_a \cos(\omega x)f(x)dx\) or \(\int^b_a \sin(\omega x)f(x)dx\) over a finite interval [a,b], where \(\omega\) and \(f\) are specified by the user. The rule evaluation component is based on the modified Clenshaw-Curtis technique

An adaptive subdivision scheme is used in connection with an extrapolation procedure, which is a modification of that in QAGS and allows the algorithm to deal with singularities in \(f(x)\).

calculates the Fourier transform \(\int^\infty_a \cos(\omega x)f(x)dx\) or \(\int^\infty_a \sin(\omega x)f(x)dx\) for user-provided \(\omega\) and \(f\). The procedure of QAWO is applied on successive finite intervals, and convergence acceleration by means of the \(\varepsilon\)-algorithm is applied to the series of integral approximations.

approximate \(\int^b_a w(x)f(x)dx\), with \(a < b\) where \(w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)\) with \(\alpha,\beta > -1\), where \(v(x)\) may be one of the following functions: \(1\), \(\log(x-a)\), \(\log(b-x)\), \(\log(x-a)\log(b-x)\).

The user specifies \(\alpha\), \(\beta\) and the type of the function \(v\). A globally adaptive subdivision strategy is applied, with modified Clenshaw-Curtis integration on those subintervals which contain a or b.

compute \(\int^b_a f(x) / (x-c)dx\) where the integral must be interpreted as a Cauchy principal value integral, for user specified \(c\) and \(f\). The strategy is globally adaptive. Modified Clenshaw-Curtis integration is used on those intervals containing the point \(x = c\).

Integration of Complex Function of a Real Variable

A complex valued function, \(f\), of a real variable can be written as \(f = g + ih\). Similarly, the integral of \(f\) can be written as

assuming that the integrals of \(g\) and \(h\) exist over the interval \([a,b]\) [2]. Therefore, quad integrates complex-valued functions by integrating the real and imaginary components separately.

Piessens, Robert; de Doncker-Kapenga, Elise; Überhuber, Christoph W.; Kahaner, David (1983). QUADPACK: A subroutine package for automatic integration. Springer-Verlag. ISBN 978-3-540-12553-2.

McCullough, Thomas; Phillips, Keith (1973). Foundations of Analysis in the Complex Plane. Holt Rinehart Winston. ISBN 0-03-086370-8

Calculate \(\int^4_0 x^2 dx\) and compare with an analytic result

Calculate \(\int^\infty_0 e^{-x} dx\)

Calculate \(\int^1_0 a x \,dx\) for \(a = 1, 3\)

Calculate \(\int^1_0 x^2 + y^2 dx\) with ctypes, holding y parameter as 1:

Be aware that pulse shapes and other sharp features as compared to the size of the integration interval may not be integrated correctly using this method. A simplified example of this limitation is integrating a y-axis reflected step function with many zero values within the integrals bounds.

**Examples:**

Example 1 (unknown):
```unknown
double func(double x)
double func(double x, void *user_data)
double func(int n, double *xx)
double func(int n, double *xx, void *user_data)
```

Example 2 (python):
```python
>>> from scipy import integrate
>>> import numpy as np
>>> x2 = lambda x: x**2
>>> integrate.quad(x2, 0, 4)
(21.333333333333332, 2.3684757858670003e-13)
>>> print(4**3 / 3.)  # analytical result
21.3333333333
```

Example 3 (unknown):
```unknown
>>> invexp = lambda x: np.exp(-x)
>>> integrate.quad(invexp, 0, np.inf)
(1.0, 5.842605999138044e-11)
```

Example 4 (unknown):
```unknown
>>> f = lambda x, a: a*x
>>> y, err = integrate.quad(f, 0, 1, args=(1,))
>>> y
0.5
>>> y, err = integrate.quad(f, 0, 1, args=(3,))
>>> y
1.5
```

---

## convolve#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html

**Contents:**
- convolve#

Multidimensional convolution.

The array is convolved with the given kernel.

Array of weights, same number of dimensions as input

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the right, and negative ones to the left. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for mode or origin must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

The result of convolution of input with weights.

Correlate an image with a kernel.

Each value in result is \(C_i = \sum_j{I_{i+k-j} W_j}\), where W is the weights kernel, j is the N-D spatial index over \(W\), I is the input and k is the coordinate of the center of W, specified by origin in the input parameters.

Perhaps the simplest case to understand is mode='constant', cval=0.0, because in this case borders (i.e., where the weights kernel, centered on any one value, extends beyond an edge of input) are treated as zeros.

Setting cval=1.0 is equivalent to padding the outer edge of input with 1.0’s (and then extracting only the original region of the result).

With mode='reflect' (the default), outer values are reflected at the edge of input to fill in missing values.

This includes diagonally at the corners.

With mode='nearest', the single nearest value in to an edge in input is repeated as many times as needed to match the overlapping weights.

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> a = np.array([[1, 2, 0, 0],
...               [5, 3, 0, 4],
...               [0, 0, 0, 7],
...               [9, 3, 0, 0]])
>>> k = np.array([[1,1,1],[1,1,0],[1,0,0]])
>>> from scipy import ndimage
>>> ndimage.convolve(a, k, mode='constant', cval=0.0)
array([[11, 10,  7,  4],
       [10,  3, 11, 11],
       [15, 12, 14,  7],
       [12,  3,  7,  0]])
```

Example 2 (json):
```json
>>> ndimage.convolve(a, k, mode='constant', cval=1.0)
array([[13, 11,  8,  7],
       [11,  3, 11, 14],
       [16, 12, 14, 10],
       [15,  6, 10,  5]])
```

Example 3 (json):
```json
>>> b = np.array([[2, 0, 0],
...               [1, 0, 0],
...               [0, 0, 0]])
>>> k = np.array([[0,1,0], [0,1,0], [0,1,0]])
>>> ndimage.convolve(b, k, mode='reflect')
array([[5, 0, 0],
       [3, 0, 0],
       [1, 0, 0]])
```

Example 4 (json):
```json
>>> k = np.array([[1,0,0],[0,1,0],[0,0,1]])
>>> ndimage.convolve(b, k)
array([[4, 2, 0],
       [3, 2, 0],
       [1, 1, 0]])
```

---

## irfftn#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfftn.html

**Contents:**
- irfftn#

Computes the inverse of rfftn

This function computes the inverse of the N-D discrete Fourier Transform for real input over any number of axes in an M-D array by means of the Fast Fourier Transform (FFT). In other words, irfftn(rfftn(x), x.shape) == x to within numerical accuracy. (The a.shape is necessary like len(a) is for irfft, and for the same reason.)

The input should be ordered in the same way as is returned by rfftn, i.e., as for irfft for the final transformation axis, and as for ifftn along all the other axes.

Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). s is also the number of input points used along this axis, except for the last axis, where s[-1]//2+1 points of the input are used. Along any axis, if the shape indicated by s is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros. If s is not given, the shape of the input along the axes specified by axes is used. Except for the last axis which is taken to be 2*(m-1), where m is the length of the input along that axis.

Axes over which to compute the inverse FFT. If not given, the last len(s) axes are used, or all axes if s is also not specified.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axes indicated by axes, or by a combination of s or x, as explained in the parameters section above. The length of each transformed axis is as given by the corresponding element of s, or the length of the input in every axis except for the last one if s is not given. In the final transformed axis the length of the output when s is not given is 2*(m-1), where m is the length of the final transformed axis of the input. To get an odd number of output points in the final axis, s must be specified.

If s and axes have different length.

If an element of axes is larger than the number of axes of x.

The forward N-D FFT of real input, of which ifftn is the inverse.

The 1-D FFT, with definitions and conventions used.

The inverse of the 1-D FFT of real input.

The inverse of the 2-D FFT of real input.

See fft for definitions and conventions used.

See rfft for definitions and conventions used for real input.

The default value of s assumes an even output length in the final transformation axis. When performing the final complex to real transformation, the Hermitian symmetry requires that the last imaginary component along that axis must be 0 and so it is ignored. To avoid losing information, the correct length of the real input must be given.

**Examples:**

Example 1 (json):
```json
>>> import scipy.fft
>>> import numpy as np
>>> x = np.zeros((3, 2, 2))
>>> x[0, 0, 0] = 3 * 2 * 2
>>> scipy.fft.irfftn(x)
array([[[1.,  1.],
        [1.,  1.]],
       [[1.,  1.],
        [1.,  1.]],
       [[1.,  1.],
        [1.,  1.]]])
```

---

## eig#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html

**Contents:**
- eig#

Solve an ordinary or generalized eigenvalue problem of a square matrix.

Find eigenvalues w and right or left eigenvectors of a general matrix:

where .H is the Hermitian conjugation.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

A complex or real matrix whose eigenvalues and eigenvectors will be computed.

Right-hand side matrix in a generalized eigenvalue problem. Default is None, identity matrix is assumed.

Whether to calculate and return left eigenvectors. Default is False.

Whether to calculate and return right eigenvectors. Default is True.

Whether to overwrite a; may improve performance. Default is False.

Whether to overwrite b; may improve performance. Default is False.

Whether to check that the input matrices contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

If True, return the eigenvalues in homogeneous coordinates. In this case w is a (2, M) array so that:

The eigenvalues, each repeated according to its multiplicity. The shape is (M,) unless homogeneous_eigvals=True.

The left eigenvector corresponding to the eigenvalue w[i] is the column vl[:,i]. Only returned if left=True. The left eigenvector is not normalized.

The normalized right eigenvector corresponding to the eigenvalue w[i] is the column vr[:,i]. Only returned if right=True.

If eigenvalue computation does not converge.

eigenvalues of general arrays

Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.

eigenvalues and right eigenvectors for symmetric/Hermitian band matrices

eigenvalues and right eiegenvectors for symmetric/Hermitian tridiagonal matrices

**Examples:**

Example 1 (unknown):
```unknown
a   vr[:,i] = w[i]        b   vr[:,i]
a.H vl[:,i] = w[i].conj() b.H vl[:,i]
```

Example 2 (unknown):
```unknown
w[1,i] a vr[:,i] = w[0,i] b vr[:,i]
```

Example 3 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> a = np.array([[0., -1.], [1., 0.]])
>>> linalg.eigvals(a)
array([0.+1.j, 0.-1.j])
```

Example 4 (unknown):
```unknown
>>> b = np.array([[0., 1.], [1., 1.]])
>>> linalg.eigvals(a, b)
array([ 1.+0.j, -1.+0.j])
```

---

## block_diag#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html

**Contents:**
- block_diag#

Create a block diagonal array from provided arrays.

For example, given 2-D inputs A, B and C, the output will have these arrays arranged on the diagonal:

Input arrays. A 1-D array or array_like sequence of length n is treated as a 2-D array with shape (1, n). Any dimensions before the last two are treated as batch dimensions; see Batched Linear Operations.

Array with A, B, C, … on the diagonal of the last two dimensions. D has the same dtype as the result type of the inputs.

If all the input arrays are square, the output is known as a block diagonal matrix.

Empty sequences (i.e., array-likes of zero size) will not be ignored. Noteworthy, both [] and [[]] are treated as matrices with shape (1,0).

**Examples:**

Example 1 (json):
```json
[[A, 0, 0],
 [0, B, 0],
 [0, 0, C]]
```

Example 2 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import block_diag
>>> A = [[1, 0],
...      [0, 1]]
>>> B = [[3, 4, 5],
...      [6, 7, 8]]
>>> C = [[7]]
>>> P = np.zeros((2, 0), dtype='int32')
>>> block_diag(A, B, C)
array([[1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 3, 4, 5, 0],
       [0, 0, 6, 7, 8, 0],
       [0, 0, 0, 0, 0, 7]])
>>> block_diag(A, P, B, C)
array([[1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 3, 4, 5, 0],
       [0, 0, 6, 7, 8, 0],
       [0, 0, 0, 0, 0, 7]])
>>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  2.,  3.,  0.,  0.],
       [ 0.,  0.,  0.,  4.,  5.],
       [ 0.,  0.,  0.,  6.,  7.]])
```

---

## Window functions (scipy.signal.windows)#

**URL:** https://docs.scipy.org/doc/scipy/reference/signal.windows.html

**Contents:**
- Window functions (scipy.signal.windows)#

The suite of window functions for filtering and spectral estimation.

get_window(window, Nx[, fftbins, xp, device])

Return a window of a given length and type.

barthann(M[, sym, xp, device])

Return a modified Bartlett-Hann window.

bartlett(M[, sym, xp, device])

Return a Bartlett window.

blackman(M[, sym, xp, device])

Return a Blackman window.

blackmanharris(M[, sym, xp, device])

Return a minimum 4-term Blackman-Harris window.

bohman(M[, sym, xp, device])

Return a Bohman window.

boxcar(M[, sym, xp, device])

Return a boxcar or rectangular window.

chebwin(M, at[, sym, xp, device])

Return a Dolph-Chebyshev window.

cosine(M[, sym, xp, device])

Return a window with a simple cosine shape.

dpss(M, NW[, Kmax, sym, norm, ...])

Compute the Discrete Prolate Spheroidal Sequences (DPSS).

exponential(M[, center, tau, sym, xp, device])

Return an exponential (or Poisson) window.

flattop(M[, sym, xp, device])

Return a flat top window.

gaussian(M, std[, sym, xp, device])

Return a Gaussian window.

general_cosine(M, a[, sym])

Generic weighted sum of cosine terms window

general_gaussian(M, p, sig[, sym, xp, device])

Return a window with a generalized Gaussian shape.

general_hamming(M, alpha[, sym, xp, device])

Return a generalized Hamming window.

hamming(M[, sym, xp, device])

Return a Hamming window.

hann(M[, sym, xp, device])

Return a Hann window.

kaiser(M, beta[, sym, xp, device])

Return a Kaiser window.

kaiser_bessel_derived(M, beta, *[, sym, xp, ...])

Return a Kaiser-Bessel derived window.

lanczos(M, *[, sym, xp, device])

Return a Lanczos window also known as a sinc window.

nuttall(M[, sym, xp, device])

Return a minimum 4-term Blackman-Harris window according to Nuttall.

parzen(M[, sym, xp, device])

Return a Parzen window.

taylor(M[, nbar, sll, norm, sym, xp, device])

Return a Taylor window.

triang(M[, sym, xp, device])

Return a triangular window.

tukey(M[, alpha, sym, xp, device])

Return a Tukey window, also known as a tapered cosine window.

---

## make_splrep#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html

**Contents:**
- make_splrep#

Create a smoothing B-spline function with bounded error, minimizing derivative jumps.

Given the set of data points (x[i], y[i]), determine a smooth spline approximation of degree k on the interval xb <= x <= xe.

The data points defining a curve y = f(x).

Strictly positive 1D array of weights, of the same length as x and y. The weights are used in computing the weighted least-squares spline fit. If the errors in the y values have standard-deviation given by the vector d, then w should be 1/d. Default is np.ones(m).

The interval to fit. If None, these default to x[0] and x[-1], respectively.

The degree of the spline fit. It is recommended to use cubic splines, k=3, which is the default. Even values of k should be avoided, especially with small s values.

The smoothing condition. The amount of smoothness is determined by satisfying the LSQ (least-squares) constraint:

where g(x) is the smoothed fit to (x, y). The user can use s to control the tradeoff between closeness to data and smoothness of fit. Larger s means more smoothing while smaller values of s indicate less smoothing. Recommended values of s depend on the weights, w. If the weights represent the inverse of the standard deviation of y, then a good s value should be found in the range (m-sqrt(2*m), m+sqrt(2*m)) where m is the number of datapoints in x, y, and w. Default is s = 0.0, i.e. interpolation.

The spline knots. If None (default), the knots will be constructed automatically. There must be at least 2*k + 2 and at most m + k + 1 knots.

The target length of the knot vector. Should be between 2*(k + 1) (the minimum number of knots for a degree-k spline), and m + k + 1 (the number of knots of the interpolating spline). The actual number of knots returned by this routine may be slightly larger than nest. Default is None (no limit, add up to m + k + 1 knots).

For s=0, spl(x) == y. For non-zero values of s the spl represents the smoothed approximation to (x, y), generally with fewer knots.

is used under the hood for generating the knots

the analog of this routine for parametric curves

construct an interpolating spline (s = 0)

construct the least-squares spline given the knot vector

a FITPACK analog of this routine

This routine constructs the smoothing spline function, \(g(x)\), to minimize the sum of jumps, \(D_j\), of the k-th derivative at the internal knots (\(x_b < t_i < x_e\)), where

Specifically, the routine constructs the spline function \(g(x)\) which minimizes

where \(s > 0\) is the input parameter.

In other words, we balance maximizing the smoothness (measured as the jumps of the derivative, the first criterion), and the deviation of \(g(x_j)\) from the data \(y_j\) (the second criterion).

Note that the summation in the second criterion is over all data points, and in the first criterion it is over the internal spline knots (i.e. those with xb < t[i] < xe). The spline knots are in general a subset of data, see generate_knots for details.

Also note the difference of this routine to make_lsq_spline: the latter routine does not consider smoothness and simply solves a least-squares problem

for a spline function \(g(x)\) with a _fixed_ knot vector t.

Added in version 1.15.0.

P. Dierckx, “Algorithms for smoothing data with periodic and parametric splines, Computer Graphics and Image Processing”, 20 (1982) 171-184.

P. Dierckx, “Curve and surface fitting with splines”, Monographs on Numerical Analysis, Oxford University Press, 1993.

**Examples:**

Example 1 (unknown):
```unknown
sum((w * (g(x)  - y))**2 ) <= s
```

---

## Statistics (scipy.stats)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/stats.html

**Contents:**
- Statistics (scipy.stats)#

In this tutorial, we discuss many, but certainly not all, features of scipy.stats. The intention here is to provide a user with a working knowledge of this package. We refer to the reference manual for further details.

Note: This documentation is work in progress.

Sample statistics and hypothesis tests

---

## Low-level BLAS functions (scipy.linalg.blas)#

**URL:** https://docs.scipy.org/doc/scipy/reference/linalg.blas.html

**Contents:**
- Low-level BLAS functions (scipy.linalg.blas)#
- Finding functions#
- BLAS Level 1 functions#
- BLAS Level 2 functions#
- BLAS Level 3 functions#

This module contains low-level functions from the BLAS library.

Added in version 0.12.0.

The common overwrite_<> option in many routines, allows the input arrays to be overwritten to avoid extra memory allocation. However this requires the array to satisfy two conditions which are memory order and the data type to match exactly the order and the type expected by the routine.

As an example, if you pass a double precision float array to any S.... routine which expects single precision arguments, f2py will create an intermediate array to match the argument types and overwriting will be performed on that intermediate array.

Similarly, if a C-contiguous array is passed, f2py will pass a FORTRAN-contiguous array internally. Please make sure that these details are satisfied. More information can be found in the f2py documentation.

These functions do little to no error checking. It is possible to cause crashes by mis-using them, so prefer using the higher-level routines in scipy.linalg.

get_blas_funcs(names[, arrays, dtype, ilp64])

Return available BLAS function objects from names.

find_best_blas_type([arrays, dtype])

Find best-matching BLAS/LAPACK type.

sasum(x,[n,offx,incx])

saxpy(x,y,[n,a,offx,incx,offy,incy])

scasum(x,[n,offx,incx])

scnrm2(x,[n,offx,incx])

scopy(x,y,[n,offx,incx,offy,incy])

sdot(x,y,[n,offx,incx,offy,incy])

snrm2(x,[n,offx,incx])

srotmg(d1, d2, x1, y1)

sscal(a,x,[n,offx,incx])

sswap(x,y,[n,offx,incx,offy,incy])

dasum(x,[n,offx,incx])

daxpy(x,y,[n,a,offx,incx,offy,incy])

dcopy(x,y,[n,offx,incx,offy,incy])

ddot(x,y,[n,offx,incx,offy,incy])

dnrm2(x,[n,offx,incx])

drotmg(d1, d2, x1, y1)

dscal(a,x,[n,offx,incx])

dswap(x,y,[n,offx,incx,offy,incy])

dzasum(x,[n,offx,incx])

dznrm2(x,[n,offx,incx])

icamax(x,[n,offx,incx])

idamax(x,[n,offx,incx])

isamax(x,[n,offx,incx])

izamax(x,[n,offx,incx])

caxpy(x,y,[n,a,offx,incx,offy,incy])

ccopy(x,y,[n,offx,incx,offy,incy])

cdotc(x,y,[n,offx,incx,offy,incy])

cdotu(x,y,[n,offx,incx,offy,incy])

cscal(a,x,[n,offx,incx])

csscal(a,x,[n,offx,incx,overwrite_x])

cswap(x,y,[n,offx,incx,offy,incy])

zaxpy(x,y,[n,a,offx,incx,offy,incy])

zcopy(x,y,[n,offx,incx,offy,incy])

zdotc(x,y,[n,offx,incx,offy,incy])

zdotu(x,y,[n,offx,incx,offy,incy])

zdscal(a,x,[n,offx,incx,overwrite_x])

zscal(a,x,[n,offx,incx])

zswap(x,y,[n,offx,incx,offy,incy])

sspr(n,alpha,x,ap,[incx,offx,lower,overwrite_ap])

ssyr(alpha,x,[lower,incx,offx,n,a,overwrite_a])

dspr(n,alpha,x,ap,[incx,offx,lower,overwrite_ap])

dsyr(alpha,x,[lower,incx,offx,n,a,overwrite_a])

cher(alpha,x,[lower,incx,offx,n,a,overwrite_a])

chpr(n,alpha,x,ap,[incx,offx,lower,overwrite_ap])

cspr(n,alpha,x,ap,[incx,offx,lower,overwrite_ap])

csyr(alpha,x,[lower,incx,offx,n,a,overwrite_a])

zher(alpha,x,[lower,incx,offx,n,a,overwrite_a])

zhpr(n,alpha,x,ap,[incx,offx,lower,overwrite_ap])

zspr(n,alpha,x,ap,[incx,offx,lower,overwrite_ap])

zsyr(alpha,x,[lower,incx,offx,n,a,overwrite_a])

ssymm(alpha,a,b,[beta,c,side,lower,overwrite_c])

ssyrk(alpha,a,[beta,c,trans,lower,overwrite_c])

dsymm(alpha,a,b,[beta,c,side,lower,overwrite_c])

dsyrk(alpha,a,[beta,c,trans,lower,overwrite_c])

chemm(alpha,a,b,[beta,c,side,lower,overwrite_c])

cherk(alpha,a,[beta,c,trans,lower,overwrite_c])

csymm(alpha,a,b,[beta,c,side,lower,overwrite_c])

csyrk(alpha,a,[beta,c,trans,lower,overwrite_c])

zhemm(alpha,a,b,[beta,c,side,lower,overwrite_c])

zherk(alpha,a,[beta,c,trans,lower,overwrite_c])

zsymm(alpha,a,b,[beta,c,side,lower,overwrite_c])

zsyrk(alpha,a,[beta,c,trans,lower,overwrite_c])

---

## Sparse Arrays (scipy.sparse)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/sparse.html

**Contents:**
- Sparse Arrays (scipy.sparse)#
- Introduction#
- Getting started with sparse arrays#
- Understanding sparse array formats#
- Sparse arrays, implicit zeros, and duplicates#
- Canonical formats#
- Next steps with sparse arrays#

scipy.sparse and its submodules provide tools for working with sparse arrays. Sparse arrays are arrays where only a few locations in the array have any data, most of the locations are considered as “empty”. Sparse arrays are useful because they allow for simpler, faster, and/or less memory-intensive algorithms for linear algebra (scipy.sparse.linalg) or graph-based computations (scipy.sparse.csgraph), but they are generally less flexible for operations like slicing, reshaping, or assignment. This guide will introduce the basics of sparse arrays in scipy.sparse, explain the unique aspects of sparse data structures, and refer onward for other sections of the user guide explaining sparse linear algebra and graph methods.

Sparse arrays are a special kind of array where only a few locations in the array have data. This allows for compressed representations of the data to be used, where only the locations where data exists are recorded. There are many different sparse array formats, each of which makes a different tradeoff between compression and functionality. To start, let’s build a very simple sparse array, the Coordinate (COO) array (coo_array) and compare it to a dense array:

Note that in our dense array, we have five nonzero values. For example, 2 is at location 0,3, and 4 is at location 1,1. All of the other values are zero. The sparse array records these five values explicitly (see the 5 stored elements and shape (3, 4)), and then represents all of the remaining zeros as implicit values.

Most sparse array methods work in a similar fashion to dense array methods:

A few “extra” properties, such as .nnz which returns the number of stored values, are present on sparse arrays as well:

Most of the reduction operations, such as .mean(), .sum(), or .max() will return a numpy array when applied over an axis of the sparse array:

This is because reductions over sparse arrays are often dense.

Different kinds of sparse arrays have different capabilities. For example, COO arrays cannot be subscripted or sliced:

But, other formats, such as the Compressed Sparse Row (CSR) csr_array support slicing and element indexing:

Sometimes, scipy.sparse will return a different sparse matrix format than the input sparse matrix format. For example, the dot product of two sparse arrays in COO format will be a CSR format array:

This change occurs because scipy.sparse will change the format of input sparse arrays in order to use the most efficient computational method.

The scipy.sparse module contains the following formats, each with their own distinct advantages and disadvantages:

Block Sparse Row (BSR) arrays scipy.sparse.bsr_array, which are most appropriate when the parts of the array with data occur in contiguous blocks.

Coordinate (COO) arrays scipy.sparse.coo_array, which provide a simple way to construct sparse arrays and modify them in place. COO can also be quickly converted into other formats, such CSR, CSC, or BSR.

Compressed Sparse Row (CSR) arrays scipy.sparse.csr_array, which are most useful for fast arithmetic, vector products, and slicing by row.

Compressed Sparse Column (CSC) arrays scipy.sparse.csc_array, which are most useful for fast arithmetic, vector products, and slicing by column.

Diagonal (DIA) arrays scipy.sparse.dia_array, which are useful for efficient storage and fast arithmetic so long as the data primarily occurs along diagonals of the array.

Dictionary of Keys (DOK) arrays scipy.sparse.dok_array, which are useful for fast construction and single-element access.

List of Lists (LIL) arrays scipy.sparse.lil_array, which are useful for fast construction and modification of sparse arrays.

More information on the strengths and weaknesses of each of the sparse array formats can be found in their documentation.

All formats of scipy.sparse arrays can be constructed directly from a numpy.ndarray. However, some sparse formats can be constructed in different ways, too. Each sparse array format has different strengths, and these strengths are documented in each class. For example, one of the most common methods for constructing sparse arrays is to build a sparse array from the individual row, column, and data values. For our array from before:

The row, column, and data arrays describe the rows, columns, and values where our sparse array has entries:

Using these, we can now define a sparse array without building a dense array first:

Different classes have different constructors, but the scipy.sparse.csr_array, scipy.sparse.csc_array, and scipy.sparse.coo_array allow for this style of construction.

Sparse arrays are useful because they represent much of their values implicitly, without storing an actual placeholder value. In scipy.sparse, the value used to represent “no data” is an implicit zero. This can be confusing when explicit zeros are required. For example, in graph methods from scipy.sparse.csgraph, we often need to be able to distinguish between (A) a link connecting nodes i and j with zero weight and (B) no link between i and j. Sparse matrices can do this, so long as we keep the explicit and implicit zeros in mind.

For example, in our previous csr array, we could include an explicit zero by including it in the data list. Let’s treat the final entry in the array at the bottom row and last column as an explicit zero:

Then, our sparse array will have six stored elements, not five:

The “extra” element is our explicit zero. The two are still identical when converted back into a dense array, because dense arrays represent everything explicitly:

But, for sparse arithmetic, linear algebra, and graph methods, the value at 2,3 will be considered an explicit zero. To remove this explicit zero, we can use the csr.eliminate_zeros() method. This operates on the sparse array in place, and removes any zero-value stored elements:

Before csr.eliminate_zeros(), there were six stored elements. After, there are only five stored elements.

Another point of complication arises from how duplicates are processed when constructing a sparse array. A duplicate can occur when we have two or more entries at row,col when constructing a sparse array. This often occurs when building sparse arrays using the data, row, and col vectors. For example, we might represent our previous array with a duplicate value at 1,1:

In this case, we can see that there are two data values that correspond to the 1,1 location in our final array. scipy.sparse will store these values separately:

Note that there are six stored elements in this sparse array, despite only having five unique locations where data occurs. When these arrays are converted back to dense arrays, the duplicate values are summed. So, at location 1,1, the dense array will contain the sum of duplicate stored entries, 1 + 3:

To remove duplicate values within the sparse array itself and thus reduce the number of stored elements, we can use the .sum_duplicates() method:

Now there are only five stored elements in our sparse array, and it is identical to the array we have been working with throughout this guide:

Several sparse array formats have “canonical formats” to allow for more efficient operations. Generally these consist of added restrictions like:

No duplicate entries for any value

Classes with a canonical form include: coo_array, csr_array, csc_array, and bsr_array. See the docstrings of these classes for details on each canonical representation.

To check if an instance of these classes is in canonical form, use the .has_canonical_format attribute:

To convert an instance to canonical form, use the .sum_duplicates() method:

Sparse array types are most helpful when working with large, nearly empty arrays. Specifically, sparse linear algebra and sparse graph methods see the largest improvements in efficiency in these circumstances.

**Examples:**

Example 1 (jsx):
```jsx
>>> import scipy as sp
>>> import numpy as np
>>> dense = np.array([[1, 0, 0, 2], [0, 4, 1, 0], [0, 0, 5, 0]])
>>> sparse = sp.sparse.coo_array(dense)
>>> dense
array([[1, 0, 0, 2],
    [0, 4, 1, 0],
    [0, 0, 5, 0]])
>>> sparse
<COOrdinate sparse array of dtype 'int64'
     with 5 stored elements and shape (3, 4)>
```

Example 2 (unknown):
```unknown
>>> sparse.max()
5
>>> dense.max()
5
>>> sparse.argmax()
10
>>> dense.argmax()
10
>>> sparse.mean()
1.0833333333333333
>>> dense.mean()
1.0833333333333333
```

Example 3 (unknown):
```unknown
>>> sparse.nnz
5
```

Example 4 (unknown):
```unknown
>>> sparse.mean(axis=1)
array([0.75, 1.25, 1.25])
```

---

## sinhm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sinhm.html

**Contents:**
- sinhm#

Compute the hyperbolic matrix sine.

This routine uses expm to compute the matrix exponentials.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Hyperbolic matrix sine of A

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> s = sinhm(a)
>>> s
array([[ 10.57300653,  39.28826594],
       [ 13.09608865,  49.86127247]])
```

Example 2 (json):
```json
>>> t = tanhm(a)
>>> c = coshm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
```

---

## fiedler_companion#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.fiedler_companion.html

**Contents:**
- fiedler_companion#

Returns a Fiedler companion matrix

Given a polynomial coefficient array a, this function forms a pentadiagonal matrix with a special structure whose eigenvalues coincides with the roots of a.

1-D array of polynomial coefficients in descending order with a nonzero leading coefficient. For N < 2, an empty array is returned. N-dimensional arrays are treated as a batch: each slice along the last axis is a 1-D array of polynomial coefficients.

Resulting companion matrix. For batch input, each slice of shape (N-1, N-1) along the last two dimensions of the output corresponds with a slice of shape (N,) along the last dimension of the input.

Similar to companion, each leading coefficient along the last axis of the input should be nonzero. If the leading coefficient is not 1, other coefficients are rescaled before the array generation. To avoid numerical issues, it is best to provide a monic polynomial.

Added in version 1.3.0.

M. Fiedler, “ A note on companion matrices”, Linear Algebra and its Applications, 2003, DOI:10.1016/S0024-3795(03)00548-2

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import fiedler_companion, eigvals
>>> p = np.poly(np.arange(1, 9, 2))  # [1., -16., 86., -176., 105.]
>>> fc = fiedler_companion(p)
>>> fc
array([[  16.,  -86.,    1.,    0.],
       [   1.,    0.,    0.,    0.],
       [   0.,  176.,    0., -105.],
       [   0.,    1.,    0.,    0.]])
>>> eigvals(fc)
array([7.+0.j, 5.+0.j, 3.+0.j, 1.+0.j])
```

---

## interp1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

**Contents:**
- interp1d#

Interpolate a 1-D function (legacy).

This class is considered legacy and will no longer receive updates. While we currently have no plans to remove it, we recommend that new code uses more modern alternatives instead. For a guide to the intended replacements for interp1d see 1-D interpolation.

x and y are arrays of values used to approximate some function f: y = f(x). This class returns a function whose call method uses interpolation to find the value of new points.

A 1-D array of real values.

A N-D array of real values. The length of y along the interpolation axis must be equal to the length of x. Use the axis parameter to select correct axis. Unlike other interpolators, the default interpolation axis is the last axis of y.

Specifies the kind of interpolation as a string or as an integer specifying the order of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.

Axis in the y array corresponding to the x-coordinate values. Unlike other interpolators, defaults to axis=-1.

If True, the class makes internal copies of x and y. If False, references to x and y are used if possible. The default is to copy.

If True, a ValueError is raised any time interpolation is attempted on a value outside of the range of x (where extrapolation is necessary). If False, out of bounds values are assigned fill_value. By default, an error is raised unless fill_value="extrapolate".

if a ndarray (or float), this value will be used to fill in for requested points outside of the data range. If not provided, then the default is NaN. The array-like must broadcast properly to the dimensions of the non-interpolation axes.

If a two-element tuple, then the first element is used as a fill value for x_new < x[0] and the second element is used for x_new > x[-1]. Anything that is not a 2-element tuple (e.g., list or ndarray, regardless of shape) is taken to be a single array-like argument meant to be used for both bounds as below, above = fill_value, fill_value. Using a two-element tuple or ndarray requires bounds_error=False.

Added in version 0.17.0.

If “extrapolate”, then points outside the data range will be extrapolated.

Added in version 0.17.0.

If False, values of x can be in any order and they are sorted first. If True, x has to be an array of monotonically increasing values.

Evaluate the interpolant

Spline interpolation/smoothing based on FITPACK.

An object-oriented wrapper of the FITPACK routines.

Calling interp1d with NaNs present in input values results in undefined behaviour.

Input values x and y must be convertible to float values like int or float.

If the values in x are not unique, the resulting behavior is undefined and specific to the choice of kind, i.e., changing kind will change the behavior for duplicates.

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy import interpolate
>>> x = np.arange(0, 10)
>>> y = np.exp(-x/3.0)
>>> f = interpolate.interp1d(x, y)
```

Example 2 (unknown):
```unknown
>>> xnew = np.arange(0, 9, 0.1)
>>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
>>> plt.plot(x, y, 'o', xnew, ynew, '-')
>>> plt.show()
```

---

## hadamard#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hadamard.html

**Contents:**
- hadamard#

Construct an Hadamard matrix.

Constructs an n-by-n Hadamard matrix, using Sylvester’s construction. n must be a power of 2.

The order of the matrix. n must be a power of 2.

The data type of the array to be constructed.

Added in version 0.8.0.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import hadamard
>>> hadamard(2, dtype=complex)
array([[ 1.+0.j,  1.+0.j],
       [ 1.+0.j, -1.-0.j]])
>>> hadamard(4)
array([[ 1,  1,  1,  1],
       [ 1, -1,  1, -1],
       [ 1,  1, -1, -1],
       [ 1, -1, -1,  1]])
```

---

## convolution_matrix#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html

**Contents:**
- convolution_matrix#

Construct a convolution matrix.

Constructs the Toeplitz matrix representing one-dimensional convolution [1]. See the notes below for details.

The 1-D array to convolve. N-dimensional arrays are treated as a batch: each slice along the last axis is a 1-D array to convolve.

The number of columns in the resulting matrix. It gives the length of the input to be convolved with a. This is analogous to the length of v in numpy.convolve(a, v).

This is analogous to mode in numpy.convolve(v, a, mode). It must be one of (‘full’, ‘valid’, ‘same’). See below for how mode determines the shape of the result.

The convolution matrix whose row count k depends on mode:

For batch input, each slice of shape (k, n) along the last two dimensions of the output corresponds with a slice of shape (m,) along the last dimension of the input.

creates a Toeplitz matrix A such that A @ v is equivalent to using convolve(a, v, mode). The returned array always has n columns. The number of rows depends on the specified mode, as explained above.

In the default ‘full’ mode, the entries of A are given by:

where m = len(a). Suppose, for example, the input array is [x, y, z]. The convolution matrix has the form:

In ‘valid’ mode, the entries of A are given by:

This corresponds to a matrix whose rows are the subset of those from the ‘full’ case where all the coefficients in a are contained in the row. For input [x, y, z], this array looks like:

In the ‘same’ mode, the entries of A are given by:

The typical application of the ‘same’ mode is when one has a signal of length n (with n greater than len(a)), and the desired output is a filtered signal that is still of length n.

For input [x, y, z], this array looks like:

Added in version 1.5.0.

“Convolution”, https://en.wikipedia.org/wiki/Convolution

Compare multiplication by A with the use of numpy.convolve.

Verify that A @ x produced the same result as applying the convolution function.

For comparison to the case mode='same' shown above, here are the matrices produced by mode='full' and mode='valid' for the same coefficients and size.

**Examples:**

Example 1 (unknown):
```unknown
=======  =========================
 mode    k
=======  =========================
'full'   m + n -1
'same'   max(m, n)
'valid'  max(m, n) - min(m, n) + 1
=======  =========================
```

Example 2 (unknown):
```unknown
A = convolution_matrix(a, n, mode)
```

Example 3 (unknown):
```unknown
A[i, j] == (a[i-j] if (0 <= (i-j) < m) else 0)
```

Example 4 (json):
```json
[x, 0, 0, ..., 0, 0]
[y, x, 0, ..., 0, 0]
[z, y, x, ..., 0, 0]
...
[0, 0, 0, ..., x, 0]
[0, 0, 0, ..., y, x]
[0, 0, 0, ..., z, y]
[0, 0, 0, ..., 0, z]
```

---

## pinv#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv.html

**Contents:**
- pinv#

Compute the (Moore-Penrose) pseudo-inverse of a matrix.

Calculate a generalized inverse of a matrix using its singular-value decomposition U @ S @ V in the economy mode and picking up only the columns/rows that are associated with significant singular values.

If s is the maximum singular value of a, then the significance cut-off value is determined by atol + rtol * s. Any singular value below this value is assumed insignificant.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Matrix to be pseudo-inverted.

Absolute threshold term, default value is 0.

Added in version 1.7.0.

Relative threshold term, default value is max(M, N) * eps where eps is the machine precision value of the datatype of a.

Added in version 1.7.0.

If True, return the effective rank of the matrix.

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

The pseudo-inverse of matrix a.

The effective rank of the matrix. Returned if return_rank is True.

If SVD computation does not converge.

Moore-Penrose pseudoinverse of a hermitian matrix.

If A is invertible then the Moore-Penrose pseudoinverse is exactly the inverse of A [1]. If A is not invertible then the Moore-Penrose pseudoinverse computes the x solution to Ax = b such that ||Ax - b|| is minimized [1].

Penrose, R. (1956). On best approximate solutions of linear matrix equations. Mathematical Proceedings of the Cambridge Philosophical Society, 52(1), 17-19. doi:10.1017/S0305004100030929

Given an m x n matrix A and an n x m matrix B the four Moore-Penrose conditions are:

ABA = A (B is a generalized inverse of A),

BAB = B (A is a generalized inverse of B),

(AB)* = AB (AB is hermitian),

(BA)* = BA (BA is hermitian) [1].

Here, A* denotes the conjugate transpose. The Moore-Penrose pseudoinverse is a unique B that satisfies all four of these conditions and exists for any A. Note that, unlike the standard matrix inverse, A does not have to be a square matrix or have linearly independent columns/rows.

As an example, we can calculate the Moore-Penrose pseudoinverse of a random non-square matrix and verify it satisfies the four conditions.

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> rng = np.random.default_rng()
>>> A = rng.standard_normal((9, 6))
>>> B = linalg.pinv(A)
>>> np.allclose(A @ B @ A, A)  # Condition 1
True
>>> np.allclose(B @ A @ B, B)  # Condition 2
True
>>> np.allclose((A @ B).conj().T, A @ B)  # Condition 3
True
>>> np.allclose((B @ A).conj().T, B @ A)  # Condition 4
True
```

---

## Sparse arrays (scipy.sparse)#

**URL:** https://docs.scipy.org/doc/scipy/reference/sparse.html

**Contents:**
- Sparse arrays (scipy.sparse)#
- Submodules#
- Sparse array classes#
  - Building sparse arrays#
  - Combining arrays#
  - Sparse tools#
  - Identifying sparse arrays#
- Sparse matrix classes#
  - Building sparse matrices#
  - Identifying sparse matrices#

SciPy 2-D sparse array package for numeric data.

This package is switching to an array interface, compatible with NumPy arrays, from the older matrix interface. We recommend that you use the array objects (bsr_array, coo_array, etc.) for all new work.

When using the array interface, please note that:

x * y no longer performs matrix multiplication, but element-wise multiplication (just like with NumPy arrays). To make code work with both arrays and matrices, use x @ y for matrix multiplication.

Operations such as sum, that used to produce dense matrices, now produce arrays, whose multiplication behavior differs similarly.

Sparse arrays use array style slicing operations, returning scalars, 1D, or 2D sparse arrays. If you need 2D results, use an appropriate index. E.g. A[:, i, None] or A[:, [i]].

All index arrays for a given sparse array should be of same dtype. For example, for CSR format, indices and indptr should have the same dtype. For COO, each array in coords should have same dtype.

The construction utilities (eye, kron, random, diags, etc.) have appropriate replacements (see Building sparse arrays).

For more information see Migration from spmatrix to sparray.

Compressed sparse graph routines (scipy.sparse.csgraph)

Sparse linear algebra (scipy.sparse.linalg)

bsr_array(arg1[, shape, dtype, copy, ...])

Block Sparse Row format sparse array.

coo_array(arg1[, shape, dtype, copy, maxprint])

A sparse array in COOrdinate format.

csc_array(arg1[, shape, dtype, copy, maxprint])

Compressed Sparse Column array.

csr_array(arg1[, shape, dtype, copy, maxprint])

Compressed Sparse Row array.

dia_array(arg1[, shape, dtype, copy, maxprint])

Sparse array with DIAgonal storage.

dok_array(arg1[, shape, dtype, copy, maxprint])

Dictionary Of Keys based sparse array.

lil_array(arg1[, shape, dtype, copy, maxprint])

Row-based LIst of Lists sparse array.

This class provides a base class for all sparse arrays.

diags_array(diagonals, /, *[, offsets, ...])

Construct a sparse array from diagonals.

eye_array(m[, n, k, dtype, format])

Sparse array of chosen shape with ones on the kth diagonal and zeros elsewhere.

random_array(shape, *[, density, format, ...])

Return a sparse array of uniformly random numbers in [0, 1)

block_array(blocks, *[, format, dtype])

Build a sparse array from sparse sub-blocks

kronecker product of sparse matrices A and B

kronsum(A, B[, format])

kronecker sum of square sparse matrices A and B

block_diag(mats[, format, dtype])

Build a block diagonal sparse matrix or array from provided matrices.

Return the lower triangular portion of a sparse array or matrix

Return the upper triangular portion of a sparse array or matrix

hstack(blocks[, format, dtype])

Stack sparse matrices horizontally (column wise)

vstack(blocks[, format, dtype])

Stack sparse arrays vertically (row wise)

save_npz(file, matrix[, compressed])

Save a sparse matrix or array to a file using .npz format.

Load a sparse array/matrix from a file using .npz format.

Return the indices and values of the nonzero elements of a matrix

get_index_dtype([arrays, maxval, check_contents])

Based on input (integer) arrays a, determine a suitable index data type that can hold the data in the arrays.

safely_cast_index_arrays(A[, idx_dtype, msg])

Safely cast sparse array indices to idx_dtype.

Is x of a sparse array or sparse matrix type?

bsr_matrix(arg1[, shape, dtype, copy, ...])

Block Sparse Row format sparse matrix.

coo_matrix(arg1[, shape, dtype, copy, maxprint])

A sparse matrix in COOrdinate format.

csc_matrix(arg1[, shape, dtype, copy, maxprint])

Compressed Sparse Column matrix.

csr_matrix(arg1[, shape, dtype, copy, maxprint])

Compressed Sparse Row matrix.

dia_matrix(arg1[, shape, dtype, copy, maxprint])

Sparse matrix with DIAgonal storage.

dok_matrix(arg1[, shape, dtype, copy, maxprint])

Dictionary Of Keys based sparse matrix.

lil_matrix(arg1[, shape, dtype, copy, maxprint])

Row-based LIst of Lists sparse matrix.

This class provides a base class for all sparse matrix classes.

eye(m[, n, k, dtype, format])

Sparse matrix of chosen shape with ones on the kth diagonal and zeros elsewhere.

identity(n[, dtype, format])

Identity matrix in sparse format

diags(diagonals[, offsets, shape, format, dtype])

Construct a sparse matrix from diagonals.

spdiags(data, diags[, m, n, format])

Return a sparse matrix from diagonals.

bmat(blocks[, format, dtype])

Build a sparse array or matrix from sparse sub-blocks

random(m, n[, density, format, dtype, rng, ...])

Generate a sparse matrix of the given shape and density with randomly distributed values.

rand(m, n[, density, format, dtype, rng, ...])

Generate a sparse matrix of the given shape and density with uniformly distributed values.

Combining matrices use the same functions as for Combining arrays.

Is x of a sparse array or sparse matrix type?

Is x of a sparse matrix type?

Is x of csc_matrix type?

Is x of csr_matrix type?

Is x of a bsr_matrix type?

Is x of lil_matrix type?

Is x of dok_array type?

Is x of coo_matrix type?

Is x of dia_matrix type?

SparseEfficiencyWarning

The warning emitted when the operation is inefficient for sparse matrices.

General warning for scipy.sparse.

There are seven available sparse array types:

csc_array: Compressed Sparse Column format

csr_array: Compressed Sparse Row format

bsr_array: Block Sparse Row format

lil_array: List of Lists format

dok_array: Dictionary of Keys format

coo_array: COOrdinate format (aka IJV, triplet format)

dia_array: DIAgonal format

To construct an array efficiently, use any of coo_array, dok_array or lil_array. dok_array and lil_array support basic slicing and fancy indexing with a similar syntax to NumPy arrays. The COO format does not support indexing (yet) but can also be used to efficiently construct arrays using coord and value info.

Despite their similarity to NumPy arrays, it is strongly discouraged to use NumPy functions directly on these arrays because NumPy typically treats them as generic Python objects rather than arrays, leading to unexpected (and incorrect) results. If you do want to apply a NumPy function to these arrays, first check if SciPy has its own implementation for the given sparse array class, or convert the sparse array to a NumPy array (e.g., using the toarray method of the class) before applying the method.

All conversions among the CSR, CSC, and COO formats are efficient, linear-time operations.

To perform manipulations such as multiplication or inversion, first convert the array to either CSC or CSR format. The lil_array format is row-based, so conversion to CSR is efficient, whereas conversion to CSC is less so.

To do a vector product between a 2D sparse array and a vector use the matmul operator (i.e., @) which performs a dot product (like the dot method):

The CSR format is especially suitable for fast matrix vector products.

Construct a 1000x1000 lil_array and add some values to it:

Now convert it to CSR format and solve A x = b for x:

Convert it to a dense array and solve, and check that the result is the same:

Now we can compute norm of the error with:

It should be small :)

Construct an array in COO format:

Notice that the indices do not need to be sorted.

Duplicate (i,j) entries are summed when converting to CSR or CSC.

This is useful for constructing finite-element stiffness and mass matrices.

CSR column indices are not necessarily sorted. Likewise for CSC row indices. Use the .sorted_indices() and .sort_indices() methods when sorted indices are required (e.g., when passing data to other libraries).

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.sparse import csr_array
>>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A @ v
array([ 1, -3, -1], dtype=int64)
```

Example 2 (sql):
```sql
>>> from scipy.sparse import lil_array
>>> from scipy.sparse.linalg import spsolve
>>> from numpy.linalg import solve, norm
>>> from numpy.random import rand
```

Example 3 (json):
```json
>>> A = lil_array((1000, 1000))
>>> A[0, :100] = rand(100)
>>> A.setdiag(rand(1000))
```

Example 4 (unknown):
```unknown
>>> A = A.tocsr()
>>> b = rand(1000)
>>> x = spsolve(A, b)
```

---

## make_interp_spline#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html

**Contents:**
- make_interp_spline#

Create an interpolating B-spline with specified degree and boundary conditions.

B-spline degree. Default is cubic, k = 3.

Knots. The number of knots needs to agree with the number of data points and the number of derivatives at the edges. Specifically, nt - n must equal len(deriv_l) + len(deriv_r).

Boundary conditions. Default is None, which means choosing the boundary conditions automatically. Otherwise, it must be a length-two tuple where the first element (deriv_l) sets the boundary conditions at x[0] and the second element (deriv_r) sets the boundary conditions at x[-1]. Each of these must be an iterable of pairs (order, value) which gives the values of derivatives of specified orders at the given edge of the interpolation interval. Alternatively, the following string aliases are recognized:

equivalent to bc_type=([(1, 0.0)], [(1, 0.0)]).

"natural": The second derivatives at ends are zero. This is equivalent to bc_type=([(2, 0.0)], [(2, 0.0)]).

"not-a-knot" (default): The first and second segments are the same polynomial. This is equivalent to having bc_type=None.

"periodic": The values and the first k-1 derivatives at the ends are equivalent.

Interpolation axis. Default is 0.

Whether to check that the input arrays contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs. Default is True.

A BSpline object of the degree k and with knots t.

base class representing the B-spline objects

a cubic spline in the polynomial basis

a similar factory function for spline fitting

a wrapper over FITPACK spline fitting routines

a wrapper over FITPACK spline fitting routines

Use cubic interpolation on Chebyshev nodes:

Note that the default is a cubic spline with a not-a-knot boundary condition

Here we use a ‘natural’ spline, with zero 2nd derivatives at edges:

Interpolation of parametric curves is also supported. As an example, we compute a discretization of a snail curve in polar coordinates

Build an interpolating curve, parameterizing it by the angle

Evaluate the interpolant on a finer grid (note that we transpose the result to unpack it into a pair of x- and y-arrays)

Build a B-spline curve with 2 dimensional y

Periodic condition is satisfied because y coordinates of points on the ends are equivalent

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> def cheb_nodes(N):
...     jj = 2.*np.arange(N) + 1
...     x = np.cos(np.pi * jj / 2 / N)[::-1]
...     return x
```

Example 2 (unknown):
```unknown
>>> x = cheb_nodes(20)
>>> y = np.sqrt(1 - x**2)
```

Example 3 (sql):
```sql
>>> from scipy.interpolate import BSpline, make_interp_spline
>>> b = make_interp_spline(x, y)
>>> np.allclose(b(x), y)
True
```

Example 4 (unknown):
```unknown
>>> l, r = [(2, 0.0)], [(2, 0.0)]
>>> b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type="natural"
>>> np.allclose(b_n(x), y)
True
>>> x0, x1 = x[0], x[-1]
>>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
True
```

---

## fourier_ellipsoid#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_ellipsoid.html

**Contents:**
- fourier_ellipsoid#

Multidimensional ellipsoid Fourier filter.

The array is multiplied with the fourier transform of an ellipsoid of given sizes.

The size of the box used for filtering. If a float, size is the same for all axes. If a sequence, size has to contain one value for each axis.

If n is negative (default), then the input is assumed to be the result of a complex fft. If n is larger than or equal to zero, the input is assumed to be the result of a real fft, and n gives the length of the array before transformation along the real transform direction.

The axis of the real transform.

If given, the result of filtering the input is placed in this array.

This function is implemented for arrays of rank 1, 2, or 3.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = datasets.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_ellipsoid(input_, size=20)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
```

---

## Spatial Transformations (scipy.spatial.transform)#

**URL:** https://docs.scipy.org/doc/scipy/reference/spatial.transform.html

**Contents:**
- Spatial Transformations (scipy.spatial.transform)#
- Rotations in 3 dimensions#

This package implements various spatial transformations. For now, rotations and rigid transforms (rotations and translations) are supported.

Rigid transform in 3 dimensions.

Rotation in 3 dimensions.

Slerp(times, rotations)

Spherical Linear Interpolation of Rotations.

RotationSpline(times, rotations)

Interpolate rotations with continuous angular rate and acceleration.

---

## Clustering package (scipy.cluster)#

**URL:** https://docs.scipy.org/doc/scipy/reference/cluster.html

**Contents:**
- Clustering package (scipy.cluster)#

Clustering algorithms are useful in information theory, target detection, communications, compression, and other areas. The vq module only supports vector quantization and the k-means algorithms.

The hierarchy module provides functions for hierarchical and agglomerative clustering. Its features include generating hierarchical clusters from distance matrices, calculating statistics on clusters, cutting linkages to generate flat clusters, and visualizing clusters with dendrograms.

---

## fourier_shift#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_shift.html

**Contents:**
- fourier_shift#

Multidimensional Fourier shift filter.

The array is multiplied with the Fourier transform of a shift operation.

The size of the box used for filtering. If a float, shift is the same for all axes. If a sequence, shift has to contain one value for each axis.

If n is negative (default), then the input is assumed to be the result of a complex fft. If n is larger than or equal to zero, the input is assumed to be the result of a real fft, and n gives the length of the array before transformation along the real transform direction.

The axis of the real transform.

If given, the result of shifting the input is placed in this array.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> import numpy.fft
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = datasets.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_shift(input_, shift=200)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
```

---

## Interpolative matrix decomposition (scipy.linalg.interpolative)#

**URL:** https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html

**Contents:**
- Interpolative matrix decomposition (scipy.linalg.interpolative)#
- Routines#
- References#
- Tutorial#
  - Initializing#
  - Computing an ID#
    - From matrix entries#
    - From matrix action#
  - Reconstructing an ID#
  - Computing an SVD#

Added in version 0.13.

Changed in version 1.15.0: The underlying algorithms have been ported to Python from the original Fortran77 code. See references below for more details.

An interpolative decomposition (ID) of a matrix \(A \in \mathbb{C}^{m \times n}\) of rank \(k \leq \min \{ m, n \}\) is a factorization

where \(\Pi = [\Pi_{1}, \Pi_{2}]\) is a permutation matrix with \(\Pi_{1} \in \{ 0, 1 \}^{n \times k}\), i.e., \(A \Pi_{2} = A \Pi_{1} T\). This can equivalently be written as \(A = BP\), where \(B = A \Pi_{1}\) and \(P = [I, T] \Pi^{\mathsf{T}}\) are the skeleton and interpolation matrices, respectively.

If \(A\) does not have exact rank \(k\), then there exists an approximation in the form of an ID such that \(A = BP + E\), where \(\| E \| \sim \sigma_{k + 1}\) is on the order of the \((k + 1)\)-th largest singular value of \(A\). Note that \(\sigma_{k + 1}\) is the best possible error for a rank-\(k\) approximation and, in fact, is achieved by the singular value decomposition (SVD) \(A \approx U S V^{*}\), where \(U \in \mathbb{C}^{m \times k}\) and \(V \in \mathbb{C}^{n \times k}\) have orthonormal columns and \(S = \mathop{\mathrm{diag}} (\sigma_{i}) \in \mathbb{C}^{k \times k}\) is diagonal with nonnegative entries. The principal advantages of using an ID over an SVD are that:

it is cheaper to construct;

it preserves the structure of \(A\); and

it is more efficient to compute with in light of the identity submatrix of \(P\).

interp_decomp(A, eps_or_k[, rand, rng])

Compute ID of a matrix.

reconstruct_matrix_from_id(B, idx, proj)

Reconstruct matrix from its ID.

reconstruct_interp_matrix(idx, proj)

Reconstruct interpolation matrix from ID.

reconstruct_skel_matrix(A, k, idx)

Reconstruct skeleton matrix from ID.

id_to_svd(B, idx, proj)

svd(A, eps_or_k[, rand, rng])

Compute SVD of a matrix via an ID.

estimate_spectral_norm(A[, its, rng])

Estimate spectral norm of a matrix by the randomized power method.

estimate_spectral_norm_diff(A, B[, its, rng])

Estimate spectral norm of the difference of two matrices by the randomized power method.

estimate_rank(A, eps[, rng])

Estimate matrix rank to a specified relative precision using randomized methods.

Following support functions are deprecated and will be removed in SciPy 1.17.0:

This function, historically, used to set the seed of the randomization algorithms used in the scipy.linalg.interpolative functions written in Fortran77.

This function, historically, used to generate uniformly distributed random number for the randomization algorithms used in the scipy.linalg.interpolative functions written in Fortran77.

This module uses the algorithms found in ID software package [R5a82238cdab4-1] by Martinsson, Rokhlin, Shkolnisky, and Tygert, which is a Fortran library for computing IDs using various algorithms, including the rank-revealing QR approach of [R5a82238cdab4-2] and the more recent randomized methods described in [R5a82238cdab4-3], [R5a82238cdab4-4], and [R5a82238cdab4-5].

We advise the user to consult also the documentation for the ID package.

P.G. Martinsson, V. Rokhlin, Y. Shkolnisky, M. Tygert. “ID: a software package for low-rank approximation of matrices via interpolative decompositions, version 0.2.” http://tygert.com/id_doc.4.pdf.

H. Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. “On the compression of low rank matrices.” SIAM J. Sci. Comput. 26 (4): 1389–1404, 2005. DOI:10.1137/030602678.

E. Liberty, F. Woolfe, P.G. Martinsson, V. Rokhlin, M. Tygert. “Randomized algorithms for the low-rank approximation of matrices.” Proc. Natl. Acad. Sci. U.S.A. 104 (51): 20167–20172, 2007. DOI:10.1073/pnas.0709640104.

P.G. Martinsson, V. Rokhlin, M. Tygert. “A randomized algorithm for the decomposition of matrices.” Appl. Comput. Harmon. Anal. 30 (1): 47–68, 2011. DOI:10.1016/j.acha.2010.02.003.

F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. “A fast randomized algorithm for the approximation of matrices.” Appl. Comput. Harmon. Anal. 25 (3): 335–366, 2008. DOI:10.1016/j.acha.2007.12.002.

The first step is to import scipy.linalg.interpolative by issuing the command:

Now let’s build a matrix. For this, we consider a Hilbert matrix, which is well know to have low rank:

We can also do this explicitly via:

Note the use of the flag order='F' in numpy.empty. This instantiates the matrix in Fortran-contiguous order and is important for avoiding data copying when passing to the backend.

We then define multiplication routines for the matrix by regarding it as a scipy.sparse.linalg.LinearOperator:

This automatically sets up methods describing the action of the matrix and its adjoint on a vector.

We have several choices of algorithm to compute an ID. These fall largely according to two dichotomies:

how the matrix is represented, i.e., via its entries or via its action on a vector; and

whether to approximate it to a fixed relative precision or to a fixed rank.

We step through each choice in turn below.

In all cases, the ID is represented by three parameters:

an index array idx; and

interpolation coefficients proj.

The ID is specified by the relation np.dot(A[:,idx[:k]], proj) == A[:,idx[k:]].

We first consider a matrix given in terms of its entries.

To compute an ID to a fixed precision, type:

where eps < 1 is the desired precision.

To compute an ID to a fixed rank, use:

where k >= 1 is the desired rank.

Both algorithms use random sampling and are usually faster than the corresponding older, deterministic algorithms, which can be accessed via the commands:

Now consider a matrix given in terms of its action on a vector as a scipy.sparse.linalg.LinearOperator.

To compute an ID to a fixed precision, type:

To compute an ID to a fixed rank, use:

These algorithms are randomized.

The ID routines above do not output the skeleton and interpolation matrices explicitly but instead return the relevant information in a more compact (and sometimes more useful) form. To build these matrices, write:

for the skeleton matrix and:

for the interpolation matrix. The ID approximation can then be computed as:

This can also be constructed directly using:

without having to first compute P.

Alternatively, this can be done explicitly as well using:

An ID can be converted to an SVD via the command:

The SVD approximation is then:

The SVD can also be computed “fresh” by combining both the ID and conversion steps into one command. Following the various ID algorithms above, there are correspondingly various SVD algorithms that one can employ.

We consider first SVD algorithms for a matrix given in terms of its entries.

To compute an SVD to a fixed precision, type:

To compute an SVD to a fixed rank, use:

Both algorithms use random sampling; for the deterministic versions, issue the keyword rand=False as above.

Now consider a matrix given in terms of its action on a vector.

To compute an SVD to a fixed precision, type:

To compute an SVD to a fixed rank, use:

Several utility routines are also available.

To estimate the spectral norm of a matrix, use:

This algorithm is based on the randomized power method and thus requires only matrix-vector products. The number of iterations to take can be set using the keyword its (default: its=20). The matrix is interpreted as a scipy.sparse.linalg.LinearOperator, but it is also valid to supply it as a numpy.ndarray, in which case it is trivially converted using scipy.sparse.linalg.aslinearoperator.

The same algorithm can also estimate the spectral norm of the difference of two matrices A1 and A2 as follows:

This is often useful for checking the accuracy of a matrix approximation.

Some routines in scipy.linalg.interpolative require estimating the rank of a matrix as well. This can be done with either:

depending on the representation. The parameter eps controls the definition of the numerical rank.

Finally, the random number generation required for all randomized routines can be controlled via providing NumPy pseudo-random generators with a fixed seed. See numpy.random.Generator and numpy.random.default_rng for more details.

The above functions all automatically detect the appropriate interface and work with both real and complex data types, passing input arguments to the proper backend routine.

**Examples:**

Example 1 (typescript):
```typescript
>>> import scipy.linalg.interpolative as sli
```

Example 2 (sql):
```sql
>>> from scipy.linalg import hilbert
>>> n = 1000
>>> A = hilbert(n)
```

Example 3 (python):
```python
>>> import numpy as np
>>> n = 1000
>>> A = np.empty((n, n), order='F')
>>> for j in range(n):
...     for i in range(n):
...         A[i,j] = 1. / (i + j + 1)
```

Example 4 (sql):
```sql
>>> from scipy.sparse.linalg import aslinearoperator
>>> L = aslinearoperator(A)
```

---

## Multidimensional image processing (scipy.ndimage)#

**URL:** https://docs.scipy.org/doc/scipy/reference/ndimage.html

**Contents:**
- Multidimensional image processing (scipy.ndimage)#
- Filters#
- Fourier filters#
- Interpolation#
- Measurements#
- Morphology#

This package contains various functions for multidimensional image processing.

convolve(input, weights[, output, mode, ...])

Multidimensional convolution.

convolve1d(input, weights[, axis, output, ...])

Calculate a 1-D convolution along the given axis.

correlate(input, weights[, output, mode, ...])

Multidimensional correlation.

correlate1d(input, weights[, axis, output, ...])

Calculate a 1-D correlation along the given axis.

gaussian_filter(input, sigma[, order, ...])

Multidimensional Gaussian filter.

gaussian_filter1d(input, sigma[, axis, ...])

gaussian_gradient_magnitude(input, sigma[, ...])

Multidimensional gradient magnitude using Gaussian derivatives.

gaussian_laplace(input, sigma[, output, ...])

Multidimensional Laplace filter using Gaussian second derivatives.

generic_filter(input, function[, size, ...])

Calculate a multidimensional filter using the given function.

generic_filter1d(input, function, filter_size)

Calculate a 1-D filter along the given axis.

generic_gradient_magnitude(input, derivative)

Gradient magnitude using a provided gradient function.

generic_laplace(input, derivative2[, ...])

N-D Laplace filter using a provided second derivative function.

laplace(input[, output, mode, cval, axes])

N-D Laplace filter based on approximate second derivatives.

maximum_filter(input[, size, footprint, ...])

Calculate a multidimensional maximum filter.

maximum_filter1d(input, size[, axis, ...])

Calculate a 1-D maximum filter along the given axis.

median_filter(input[, size, footprint, ...])

Calculate a multidimensional median filter.

minimum_filter(input[, size, footprint, ...])

Calculate a multidimensional minimum filter.

minimum_filter1d(input, size[, axis, ...])

Calculate a 1-D minimum filter along the given axis.

percentile_filter(input, percentile[, size, ...])

Calculate a multidimensional percentile filter.

prewitt(input[, axis, output, mode, cval])

Calculate a Prewitt filter.

rank_filter(input, rank[, size, footprint, ...])

Calculate a multidimensional rank filter.

sobel(input[, axis, output, mode, cval])

Calculate a Sobel filter.

uniform_filter(input[, size, output, mode, ...])

Multidimensional uniform filter.

uniform_filter1d(input, size[, axis, ...])

Calculate a 1-D uniform filter along the given axis.

vectorized_filter(input, function, *[, ...])

Filter an array with a vectorized Python callable as the kernel

fourier_ellipsoid(input, size[, n, axis, output])

Multidimensional ellipsoid Fourier filter.

fourier_gaussian(input, sigma[, n, axis, output])

Multidimensional Gaussian fourier filter.

fourier_shift(input, shift[, n, axis, output])

Multidimensional Fourier shift filter.

fourier_uniform(input, size[, n, axis, output])

Multidimensional uniform fourier filter.

affine_transform(input, matrix[, offset, ...])

Apply an affine transformation.

geometric_transform(input, mapping[, ...])

Apply an arbitrary geometric transform.

map_coordinates(input, coordinates[, ...])

Map the input array to new coordinates by interpolation.

rotate(input, angle[, axes, reshape, ...])

shift(input, shift[, output, order, mode, ...])

spline_filter(input[, order, output, mode])

Multidimensional spline filter.

spline_filter1d(input[, order, axis, ...])

Calculate a 1-D spline filter along the given axis.

zoom(input, zoom[, output, order, mode, ...])

center_of_mass(input[, labels, index])

Calculate the center of mass of the values of an array at labels.

extrema(input[, labels, index])

Calculate the minimums and maximums of the values of an array at labels, along with their positions.

find_objects(input[, max_label])

Find objects in a labeled array.

histogram(input, min, max, bins[, labels, index])

Calculate the histogram of the values of an array, optionally at labels.

label(input[, structure, output])

Label features in an array.

labeled_comprehension(input, labels, index, ...)

Roughly equivalent to [func(input[labels == i]) for i in index].

maximum(input[, labels, index])

Calculate the maximum of the values of an array over labeled regions.

maximum_position(input[, labels, index])

Find the positions of the maximums of the values of an array at labels.

mean(input[, labels, index])

Calculate the mean of the values of an array at labels.

median(input[, labels, index])

Calculate the median of the values of an array over labeled regions.

minimum(input[, labels, index])

Calculate the minimum of the values of an array over labeled regions.

minimum_position(input[, labels, index])

Find the positions of the minimums of the values of an array at labels.

standard_deviation(input[, labels, index])

Calculate the standard deviation of the values of an N-D image array, optionally at specified sub-regions.

sum_labels(input[, labels, index])

Calculate the sum of the values of the array.

value_indices(arr, *[, ignore_value])

Find indices of each distinct value in given array.

variance(input[, labels, index])

Calculate the variance of the values of an N-D image array, optionally at specified sub-regions.

watershed_ift(input, markers[, structure, ...])

Apply watershed from markers using image foresting transform algorithm.

binary_closing(input[, structure, ...])

Multidimensional binary closing with the given structuring element.

binary_dilation(input[, structure, ...])

Multidimensional binary dilation with the given structuring element.

binary_erosion(input[, structure, ...])

Multidimensional binary erosion with a given structuring element.

binary_fill_holes(input[, structure, ...])

Fill the holes in binary objects.

binary_hit_or_miss(input[, structure1, ...])

Multidimensional binary hit-or-miss transform.

binary_opening(input[, structure, ...])

Multidimensional binary opening with the given structuring element.

binary_propagation(input[, structure, mask, ...])

Multidimensional binary propagation with the given structuring element.

black_tophat(input[, size, footprint, ...])

Multidimensional black tophat filter.

distance_transform_bf(input[, metric, ...])

Distance transform function by a brute force algorithm.

distance_transform_cdt(input[, metric, ...])

Distance transform for chamfer type of transforms.

distance_transform_edt(input[, sampling, ...])

Exact Euclidean distance transform.

generate_binary_structure(rank, connectivity)

Generate a binary structure for binary morphological operations.

grey_closing(input[, size, footprint, ...])

Multidimensional grayscale closing.

grey_dilation(input[, size, footprint, ...])

Calculate a greyscale dilation, using either a structuring element, or a footprint corresponding to a flat structuring element.

grey_erosion(input[, size, footprint, ...])

Calculate a greyscale erosion, using either a structuring element, or a footprint corresponding to a flat structuring element.

grey_opening(input[, size, footprint, ...])

Multidimensional grayscale opening.

iterate_structure(structure, iterations[, ...])

Iterate a structure by dilating it with itself.

morphological_gradient(input[, size, ...])

Multidimensional morphological gradient.

morphological_laplace(input[, size, ...])

Multidimensional morphological laplace.

white_tophat(input[, size, footprint, ...])

Multidimensional white tophat filter.

---

## hilbert#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hilbert.html

**Contents:**
- hilbert#

Create a Hilbert matrix of order n.

Returns the n by n array with entries h[i,j] = 1 / (i + j + 1).

The size of the array to create.

Compute the inverse of a Hilbert matrix.

Added in version 0.10.0.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import hilbert
>>> hilbert(3)
array([[ 1.        ,  0.5       ,  0.33333333],
       [ 0.5       ,  0.33333333,  0.25      ],
       [ 0.33333333,  0.25      ,  0.2       ]])
```

---

## gaussian_gradient_magnitude#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_gradient_magnitude.html

**Contents:**
- gaussian_gradient_magnitude#

Multidimensional gradient magnitude using Gaussian derivatives.

The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

The axes over which to apply the filter. If sigma or mode tuples are provided, their length must match the number of axes.

Filtered array. Has the same shape as input.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = ndimage.gaussian_gradient_magnitude(ascent, sigma=5)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## percentile_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html

**Contents:**
- percentile_filter#

Calculate a multidimensional percentile filter.

The percentile parameter may be less than zero, i.e., percentile = -20 equals percentile = 80

See footprint, below. Ignored if footprint is given.

Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function. footprint is a boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)). We adjust size to the number of dimensions of the input array, so that, if the input array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When footprint is given, size is ignored.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for size, origin, and/or mode must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

Filtered array. Has the same shape as input.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = ndimage.percentile_filter(ascent, percentile=20, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## simpson#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html

**Contents:**
- simpson#

Integrate y(x) using samples along the given axis and the composite Simpson’s rule. If x is None, spacing of dx is assumed.

Array to be integrated.

If given, the points at which y is sampled.

Spacing of integration points along axis of x. Only used when x is None. Default is 1.

Axis along which to integrate. Default is the last axis.

The estimated integral computed with the composite Simpson’s rule.

adaptive quadrature using QUADPACK

fixed-order Gaussian quadrature

integrators for sampled data

cumulative integration for sampled data

cumulative integration using Simpson’s 1/3 rule

For an odd number of samples that are equally spaced the result is exact if the function is a polynomial of order 3 or less. If the samples are not equally spaced, then the result is exact only if the function is a polynomial of order 2 or less.

Cartwright, Kenneth V. Simpson’s Rule Cumulative Integration with MS Excel and Irregularly-spaced Data. Journal of Mathematical Sciences and Mathematics Education. 12 (2): 1-9

**Examples:**

Example 1 (python):
```python
>>> from scipy import integrate
>>> import numpy as np
>>> x = np.arange(0, 10)
>>> y = np.arange(0, 10)
```

Example 2 (unknown):
```unknown
>>> integrate.simpson(y, x=x)
40.5
```

Example 3 (unknown):
```unknown
>>> y = np.power(x, 3)
>>> integrate.simpson(y, x=x)
1640.5
>>> integrate.quad(lambda x: x**3, 0, 9)[0]
1640.25
```

---

## Datasets (scipy.datasets)#

**URL:** https://docs.scipy.org/doc/scipy/reference/datasets.html

**Contents:**
- Datasets (scipy.datasets)#
- Dataset Methods#
- Utility Methods#
- Usage of Datasets#
- How dataset retrieval and storage works#

Get an 8-bit grayscale bit-depth, 512 x 512 derived image for easy use in demos.

Get a 1024 x 768, color image of a raccoon face.

Load an electrocardiogram as an example for a 1-D signal.

Utility method to download all the dataset files for scipy.datasets module.

clear_cache([datasets])

Cleans the scipy datasets cache directory.

SciPy dataset methods can be simply called as follows: '<dataset-name>()' This downloads the dataset files over the network once, and saves the cache, before returning a numpy.ndarray object representing the dataset.

Note that the return data structure and data type might be different for different dataset methods. For a more detailed example on usage, please look into the particular dataset method documentation above.

SciPy dataset files are stored within individual GitHub repositories under the SciPy GitHub organization, following a naming convention as 'dataset-<name>', for example scipy.datasets.face files live at scipy/dataset-face. The scipy.datasets submodule utilizes and depends on Pooch, a Python package built to simplify fetching data files. Pooch uses these repos to retrieve the respective dataset files when calling the dataset function.

A registry of all the datasets, essentially a mapping of filenames with their SHA256 hash and repo urls are maintained, which Pooch uses to handle and verify the downloads on function call. After downloading the dataset once, the files are saved in the system cache directory under 'scipy-data'.

Dataset cache locations may vary on different platforms.

For Linux and other Unix-like platforms:

In environments with constrained network connectivity for various security reasons or on systems without continuous internet connections, one may manually load the cache of the datasets by placing the contents of the dataset repo in the above mentioned cache directory to avoid fetching dataset errors without the internet connectivity.

**Examples:**

Example 1 (unknown):
```unknown
'~/Library/Caches/scipy-data'
```

Example 2 (unknown):
```unknown
'~/.cache/scipy-data'  # or the value of the XDG_CACHE_HOME env var, if defined
```

Example 3 (typescript):
```typescript
'C:\Users\<user>\AppData\Local\<AppAuthor>\scipy-data\Cache'
```

---

## trapezoid#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html

**Contents:**
- trapezoid#

Integrate along the given axis using the composite trapezoidal rule.

If x is provided, the integration happens in sequence along its elements - they are not sorted.

Integrate y (x) along each 1d slice on the given axis, compute \(\int y(x) dx\). When x is specified, this integrates along the parametric curve, computing \(\int_t y(t) dt = \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt\).

Input array to integrate.

The sample points corresponding to the y values. If x is None, the sample points are assumed to be evenly spaced dx apart. The default is None.

The spacing between sample points when x is None. The default is 1.

The axis along which to integrate. The default is the last axis.

Definite integral of y = n-dimensional array as approximated along a single axis by the trapezoidal rule. If y is a 1-dimensional array, then the result is a float. If n is greater than 1, then the result is an n-1 dimensional array.

Image [2] illustrates trapezoidal rule – y-axis locations of points will be taken from y array, by default x-axis distances between points will be 1.0, alternatively they can be provided with x array or with dx scalar. Return value will be equal to combined area under the red lines.

Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

Illustration image: https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

Use the trapezoidal rule on evenly spaced points:

The spacing between sample points can be selected by either the x or dx arguments:

Using a decreasing x corresponds to integrating in reverse:

More generally x is used to integrate along a parametric curve. We can estimate the integral \(\int_0^1 x^2 = 1/3\) using:

Or estimate the area of a circle, noting we repeat the sample which closes the curve:

trapezoid can be applied along a specified axis to do multiple computations in one call:

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import integrate
>>> integrate.trapezoid([1, 2, 3])
4.0
```

Example 2 (unknown):
```unknown
>>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])
8.0
>>> integrate.trapezoid([1, 2, 3], dx=2)
8.0
```

Example 3 (unknown):
```unknown
>>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])
-8.0
```

Example 4 (unknown):
```unknown
>>> x = np.linspace(0, 1, num=50)
>>> y = x**2
>>> integrate.trapezoid(y, x)
0.33340274885464394
```

---

## coshm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.coshm.html

**Contents:**
- coshm#

Compute the hyperbolic matrix cosine.

This routine uses expm to compute the matrix exponentials.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Hyperbolic matrix cosine of A

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> c = coshm(a)
>>> c
array([[ 11.24592233,  38.76236492],
       [ 12.92078831,  50.00828725]])
```

Example 2 (json):
```json
>>> t = tanhm(a)
>>> s = sinhm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
```

---

## Quasi-Monte Carlo submodule (scipy.stats.qmc)#

**URL:** https://docs.scipy.org/doc/scipy/reference/stats.qmc.html

**Contents:**
- Quasi-Monte Carlo submodule (scipy.stats.qmc)#
- Quasi-Monte Carlo#
  - Engines#
  - Helpers#
- Introduction to Quasi-Monte Carlo#
  - References#

This module provides Quasi-Monte Carlo generators and associated helper functions.

QMCEngine(d, *[, optimization, rng, seed])

A generic Quasi-Monte Carlo sampler class meant for subclassing.

Sobol(d, *[, scramble, bits, rng, ...])

Engine for generating (scrambled) Sobol' sequences.

Halton(d, *[, scramble, optimization, rng, seed])

LatinHypercube(d, *[, scramble, strength, ...])

Latin hypercube sampling (LHS).

PoissonDisk(d, *[, radius, hypersphere, ...])

Poisson disk sampling.

MultinomialQMC(pvals, n_trials, *[, engine, ...])

QMC sampling from a multinomial distribution.

MultivariateNormalQMC(mean[, cov, cov_root, ...])

QMC sampling from a multivariate Normal \(N(\mu, \Sigma)\).

discrepancy(sample, *[, iterative, method, ...])

Discrepancy of a given sample.

geometric_discrepancy(sample[, method, metric])

Discrepancy of a given sample based on its geometric properties.

update_discrepancy(x_new, sample, initial_disc)

Update the centered discrepancy with a new sample.

scale(sample, l_bounds, u_bounds, *[, reverse])

Sample scaling from unit hypercube to different bounds.

Quasi-Monte Carlo (QMC) methods [1], [2], [3] provide an \(n \times d\) array of numbers in \([0,1]\). They can be used in place of \(n\) points from the \(U[0,1]^{d}\) distribution. Compared to random points, QMC points are designed to have fewer gaps and clumps. This is quantified by discrepancy measures [4]. From the Koksma-Hlawka inequality [5] we know that low discrepancy reduces a bound on integration error. Averaging a function \(f\) over \(n\) QMC points can achieve an integration error close to \(O(n^{-1})\) for well behaved functions [2].

Most QMC constructions are designed for special values of \(n\) such as powers of 2 or large primes. Changing the sample size by even one can degrade their performance, even their rate of convergence [6]. For instance \(n=100\) points may give less accuracy than \(n=64\) if the method was designed for \(n=2^m\).

Some QMC constructions are extensible in \(n\): we can find another special sample size \(n' > n\) and often an infinite sequence of increasing special sample sizes. Some QMC constructions are extensible in \(d\): we can increase the dimension, possibly to some upper bound, and typically without requiring special values of \(d\). Some QMC methods are extensible in both \(n\) and \(d\).

QMC points are deterministic. That makes it hard to estimate the accuracy of integrals estimated by averages over QMC points. Randomized QMC (RQMC) [7] points are constructed so that each point is individually \(U[0,1]^{d}\) while collectively the \(n\) points retain their low discrepancy. One can make \(R\) independent replications of RQMC points to see how stable a computation is. From \(R\) independent values, a t-test (or bootstrap t-test [8]) then gives approximate confidence intervals on the mean value. Some RQMC methods produce a root mean squared error that is actually \(o(1/n)\) and smaller than the rate seen in unrandomized QMC. An intuitive explanation is that the error is a sum of many small ones and random errors cancel in a way that deterministic ones do not. RQMC also has advantages on integrands that are singular or, for other reasons, fail to be Riemann integrable.

(R)QMC cannot beat Bahkvalov’s curse of dimension (see [9]). For any random or deterministic method, there are worst case functions that will give it poor performance in high dimensions. A worst case function for QMC might be 0 at all n points but very large elsewhere. Worst case analyses get very pessimistic in high dimensions. (R)QMC can bring a great improvement over MC when the functions on which it is used are not worst case. For instance (R)QMC can be especially effective on integrands that are well approximated by sums of functions of some small number of their input variables at a time [10], [11]. That property is often a surprising finding about those functions.

Also, to see an improvement over IID MC, (R)QMC requires a bit of smoothness of the integrand, roughly the mixed first order derivative in each direction, \(\partial^d f/\partial x_1 \cdots \partial x_d\), must be integral. For instance, a function that is 1 inside the hypersphere and 0 outside of it has infinite variation in the sense of Hardy and Krause for any dimension \(d = 2\).

Scrambled nets are a kind of RQMC that have some valuable robustness properties [12]. If the integrand is square integrable, they give variance \(var_{SNET} = o(1/n)\). There is a finite upper bound on \(var_{SNET} / var_{MC}\) that holds simultaneously for every square integrable integrand. Scrambled nets satisfy a strong law of large numbers for \(f\) in \(L^p\) when \(p>1\). In some special cases there is a central limit theorem [13]. For smooth enough integrands they can achieve RMSE nearly \(O(n^{-3})\). See [12] for references about these properties.

The main kinds of QMC methods are lattice rules [14] and digital nets and sequences [2], [15]. The theories meet up in polynomial lattice rules [16] which can produce digital nets. Lattice rules require some form of search for good constructions. For digital nets there are widely used default constructions.

The most widely used QMC methods are Sobol’ sequences [17]. These are digital nets. They are extensible in both \(n\) and \(d\). They can be scrambled. The special sample sizes are powers of 2. Another popular method are Halton sequences [18]. The constructions resemble those of digital nets. The earlier dimensions have much better equidistribution properties than later ones. There are essentially no special sample sizes. They are not thought to be as accurate as Sobol’ sequences. They can be scrambled. The nets of Faure [19] are also widely used. All dimensions are equally good, but the special sample sizes grow rapidly with dimension \(d\). They can be scrambled. The nets of Niederreiter and Xing [20] have the best asymptotic properties but have not shown good empirical performance [21].

Higher order digital nets are formed by a digit interleaving process in the digits of the constructed points. They can achieve higher levels of asymptotic accuracy given higher smoothness conditions on \(f\) and they can be scrambled [22]. There is little or no empirical work showing the improved rate to be attained.

Using QMC is like using the entire period of a small random number generator. The constructions are similar and so therefore are the computational costs [23].

(R)QMC is sometimes improved by passing the points through a baker’s transformation (tent function) prior to using them. That function has the form \(1-2|x-1/2|\). As \(x\) goes from 0 to 1, this function goes from 0 to 1 and then back. It is very useful to produce a periodic function for lattice rules [14], and sometimes it improves the convergence rate [24].

It is not straightforward to apply QMC methods to Markov chain Monte Carlo (MCMC). We can think of MCMC as using \(n=1\) point in \([0,1]^{d}\) for very large \(d\), with ergodic results corresponding to \(d \to \infty\). One proposal is in [25] and under strong conditions an improved rate of convergence has been shown [26].

Returning to Sobol’ points: there are many versions depending on what are called direction numbers. Those are the result of searches and are tabulated. A very widely used set of direction numbers come from [27]. It is extensible in dimension up to \(d=21201\).

Owen, Art B. “Monte Carlo Book: the Quasi-Monte Carlo parts.” 2019.

Niederreiter, Harald. “Random number generation and quasi-Monte Carlo methods.” Society for Industrial and Applied Mathematics, 1992.

Dick, Josef, Frances Y. Kuo, and Ian H. Sloan. “High-dimensional integration: the quasi-Monte Carlo way.” Acta Numerica no. 22: 133, 2013.

Aho, A. V., C. Aistleitner, T. Anderson, K. Appel, V. Arnol’d, N. Aronszajn, D. Asotsky et al. “W. Chen et al.(eds.), “A Panorama of Discrepancy Theory”, Sringer International Publishing, Switzerland: 679, 2014.

Hickernell, Fred J. “Koksma-Hlawka Inequality.” Wiley StatsRef: Statistics Reference Online, 2014.

Owen, Art B. “On dropping the first Sobol’ point.” arXiv:2008.08051, 2020.

L’Ecuyer, Pierre, and Christiane Lemieux. “Recent advances in randomized quasi-Monte Carlo methods.” In Modeling uncertainty, pp. 419-474. Springer, New York, NY, 2002.

DiCiccio, Thomas J., and Bradley Efron. “Bootstrap confidence intervals.” Statistical science: 189-212, 1996.

Dimov, Ivan T. “Monte Carlo methods for applied scientists.” World Scientific, 2008.

Caflisch, Russel E., William J. Morokoff, and Art B. Owen. “Valuation of mortgage backed securities using Brownian bridges to reduce effective dimension.” Journal of Computational Finance: no. 1 27-46, 1997.

Sloan, Ian H., and Henryk Wozniakowski. “When are quasi-Monte Carlo algorithms efficient for high dimensional integrals?.” Journal of Complexity 14, no. 1 (1998): 1-33.

Owen, Art B., and Daniel Rudolf, “A strong law of large numbers for scrambled net integration.” SIAM Review, to appear.

Loh, Wei-Liem. “On the asymptotic distribution of scrambled net quadrature.” The Annals of Statistics 31, no. 4: 1282-1324, 2003.

Sloan, Ian H. and S. Joe. “Lattice methods for multiple integration.” Oxford University Press, 1994.

Dick, Josef, and Friedrich Pillichshammer. “Digital nets and sequences: discrepancy theory and quasi-Monte Carlo integration.” Cambridge University Press, 2010.

Dick, Josef, F. Kuo, Friedrich Pillichshammer, and I. Sloan. “Construction algorithms for polynomial lattice rules for multivariate integration.” Mathematics of computation 74, no. 252: 1895-1921, 2005.

Sobol’, Il’ya Meerovich. “On the distribution of points in a cube and the approximate evaluation of integrals.” Zhurnal Vychislitel’noi Matematiki i Matematicheskoi Fiziki 7, no. 4: 784-802, 1967.

Halton, John H. “On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals.” Numerische Mathematik 2, no. 1: 84-90, 1960.

Faure, Henri. “Discrepance de suites associees a un systeme de numeration (en dimension s).” Acta arithmetica 41, no. 4: 337-351, 1982.

Niederreiter, Harold, and Chaoping Xing. “Low-discrepancy sequences and global function fields with many rational places.” Finite Fields and their applications 2, no. 3: 241-273, 1996.

Hong, Hee Sun, and Fred J. Hickernell. “Algorithm 823: Implementing scrambled digital sequences.” ACM Transactions on Mathematical Software (TOMS) 29, no. 2: 95-109, 2003.

Dick, Josef. “Higher order scrambled digital nets achieve the optimal rate of the root mean square error for smooth integrands.” The Annals of Statistics 39, no. 3: 1372-1398, 2011.

Niederreiter, Harald. “Multidimensional numerical integration using pseudorandom numbers.” In Stochastic Programming 84 Part I, pp. 17-38. Springer, Berlin, Heidelberg, 1986.

Hickernell, Fred J. “Obtaining O (N-2+e) Convergence for Lattice Quadrature Rules.” In Monte Carlo and Quasi-Monte Carlo Methods 2000, pp. 274-289. Springer, Berlin, Heidelberg, 2002.

Owen, Art B., and Seth D. Tribble. “A quasi-Monte Carlo Metropolis algorithm.” Proceedings of the National Academy of Sciences 102, no. 25: 8844-8849, 2005.

Chen, Su. “Consistency and convergence rate of Markov chain quasi Monte Carlo with examples.” PhD diss., Stanford University, 2011.

Joe, Stephen, and Frances Y. Kuo. “Constructing Sobol sequences with better two-dimensional projections.” SIAM Journal on Scientific Computing 30, no. 5: 2635-2654, 2008.

---

## Extrapolation tips and tricks#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/interpolate/extrapolation_examples.html

**Contents:**
- Extrapolation tips and tricks#
- interp1d : replicate numpy.interp left and right fill values#
- CubicSpline extend the boundary conditions#
- Manually implement the asymptotics#
  - The setup#
  - Use the known asymptotics#
- Extrapolation in D > 1#

Handling of extrapolation—evaluation of the interpolators on query points outside of the domain of interpolated data—is not fully consistent among different routines in scipy.interpolate. Different interpolators use different sets of keyword arguments to control the behavior outside of the data domain: some use extrapolate=True/False/None, some allow the fill_value keyword. Refer to the API documentation for details for each specific interpolation routine.

Depending on a particular problem, the available keywords may or may not be sufficient. Special attention needs to be paid to extrapolation of non-linear interpolants. Very often the extrapolated results make less sense with increasing distance from the data domain. This is of course to be expected: an interpolant only knows the data within the data domain.

When the default extrapolated results are not adequate, users need to implement the desired extrapolation mode themselves.

In this tutorial, we consider several worked examples where we demonstrate both the use of available keywords and manual implementation of desired extrapolation modes. These examples may or may not be applicable to your particular problem; they are not necessarily best practices; and they are deliberately pared down to bare essentials needed to demonstrate the main ideas, in a hope that they serve as an inspiration for your handling of your particular problem.

TL;DR: Use fill_value=(left, right)

numpy.interp uses constant extrapolation, and defaults to extending the first and last values of the y array in the interpolation interval: the output of np.interp(xnew, x, y) is y[0] for xnew < x[0] and y[-1] for xnew > x[-1].

By default, interp1d refuses to extrapolate, and raises a ValueError when evaluated on a data point outside of the interpolation range. This can be switched off by the bounds_error=False argument: then interp1d sets the out-of-range values with the fill_value, which is nan by default.

To mimic the behavior of numpy.interp with interp1d, you can use the fact that it supports a 2-tuple as the fill_value. The tuple elements are then used to fill for xnew < min(x) and x > max(x), respectively. For multidimensional y, these elements must have the same shape as y or be broadcastable to it.

CubicSpline needs two extra boundary conditions, which are controlled by the bc_type parameter. This parameter can either list explicit values of derivatives at the edges, or use helpful aliases. For instance, bc_type="clamped" sets the first derivatives to zero, bc_type="natural" sets the second derivatives to zero (two other recognized string values are “periodic” and “not-a-knot”)

While the extrapolation is controlled by the boundary condition, the relation is not very intuitive. For instance, one can expect that for bc_type="natural", the extrapolation is linear. This expectation is too strong: each boundary condition sets the derivatives at a single point, at the boundary only. Extrapolation is done from the first and last polynomial pieces, which — for a natural spline — is a cubic with a zero second derivative at a given point.

One other way of seeing why this expectation is too strong is to consider a dataset with only three data points, where the spline has two polynomial pieces. To extrapolate linearly, this expectation implies that both of these pieces are linear. But then, two linear pieces cannot match at a middle point with a continuous 2nd derivative! (Unless of course, if all three data points actually lie on a single straight line).

To illustrate the behavior we consider a synthetic dataset and compare several boundary conditions:

It is clearly seen that the natural spline does have the zero second derivative at the boundaries, but extrapolation is non-linear. bc_type="clamped" shows a similar behavior: first derivatives are only equal to zero exactly at the boundary. In all cases, extrapolation is done by extending the first and last polynomial pieces of the spline, whatever they happen to be.

One possible way to force the extrapolation is to extend the interpolation domain to add first and last polynomial pieces which have desired properties.

Here we use extend method of the CubicSpline superclass, PPoly, to add two extra breakpoints and to make sure that the additional polynomial pieces maintain the values of the derivatives. Then the extrapolation proceeds using these two additional intervals.

The previous trick of extending the interpolation domain relies on the CubicSpline.extend method. A somewhat more general alternative is to implement a wrapper which handles the out-of-bounds behavior explicitly. Let us consider a worked example.

Suppose we want to solve at a given value of \(a\) the equation

(One application where these kinds of equations appear is solving for energy levels of a quantum particle). For simplicity, let’s only consider \(x\in (0, \pi/2)\).

Solving this equation once is straightforward:

However, if we need to solve it multiple times (e.g. to find a series of roots due to periodicity of the tan function), repeated calls to scipy.optimize.brentq become prohibitively expensive.

To circumvent this difficulty, we tabulate \(y = ax - 1/\tan{x}\) and interpolate it on the tabulated grid. In fact, we will use the inverse interpolation: we interpolate the values of \(x\) versus \(у\). This way, solving the original equation becomes simply an evaluation of the interpolated function at zero \(y\) argument.

To improve the interpolation accuracy we will use the knowledge of the derivatives of the tabulated function. We will use BPoly.from_derivatives to construct a cubic interpolant (equivalently, we could have used CubicHermiteSpline)

Note that for \(a=3\), spl(0) agrees with the brentq call above, while for \(a = 93\), the difference is substantial. The reason the procedure starts failing for large \(a\) is that the straight line \(y = ax\) tends towards the vertical axis, and the root of the original equation tends towards \(x=0\). Since we tabulated the original function at a finite grid, spl(0) involves extrapolation for too-large values of \(a\). Relying on extrapolation is prone to losing accuracy and is best avoided.

Looking at the original equation, we note that for \(x\to 0\), \(\tan(x) = x + O(x^3)\), and the original equation becomes

so that \(x_0 \approx 1/\sqrt{a}\) for \(a \gg 1\).

We will use this to cook up a class which switches from interpolation to using this known asymptotic behavior for out-of-range data. A bare-bones implementation may look like this

which differs from the extrapolated result and agrees with the brentq call.

Note that this implementation is intentionally pared down. From the API perspective, you may want to instead implement the __call__ method so that the full dependence of x on y is available. From the numerical perspective, more work is needed to make sure that the switch between interpolation and asymptotics occurs deep enough into the asymptotic regime, so that the resulting function is smooth enough at the switch-over point.

Also in this example we artificially limited the problem to only consider a single periodicity interval of the tan function, and only dealt with \(a > 0\). For negative values of \(a\), we would need to implement the other asymptotics, for \(x\to \pi\).

However the basic idea is the same.

The basic idea of implementing extrapolation manually in a wrapper class or function can be easily generalized to higher dimensions. As an example, we consider a C1-smooth interpolation problem of 2D data using CloughTocher2DInterpolator. By default, it fills the out of bounds values with nans, and we want to instead use for each query point the value of its nearest neighbor.

Since CloughTocher2DInterpolator accepts either 2D data or a Delaunay triangulation of the data points, the efficient way of finding nearest neighbors of query points would be to construct the triangulation (using scipy.spatial tools) and use it to find nearest neighbors on the convex hull of the data.

We will instead use a simpler, naive method and rely on looping over the whole dataset using NumPy broadcasting.

**Examples:**

Example 1 (sql):
```sql
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.linspace(0, 1.5*np.pi, 11)
y = np.column_stack((np.cos(x), np.sin(x)))   # y.shape is (11, 2)

func = interp1d(x, y,
                axis=0,  # interpolate along columns
                bounds_error=False,
                kind='linear',
                fill_value=(y[0], y[-1]))
xnew = np.linspace(-np.pi, 2.5*np.pi, 51)
ynew = func(xnew)

fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.plot(xnew, ynew[:, 0])
ax1.plot(x, y[:, 0], 'o')

ax2.plot(xnew, ynew[:, 1])
ax2.plot(x, y[:, 1], 'o')
plt.tight_layout()
```

Example 2 (python):
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

xs = [1, 2, 3, 4, 5, 6, 7, 8]
ys = [4.5, 3.6, 1.6, 0.0, -3.3, -3.1, -1.8, -1.7]

notaknot = CubicSpline(xs, ys, bc_type='not-a-knot')
natural = CubicSpline(xs, ys, bc_type='natural')
clamped = CubicSpline(xs, ys, bc_type='clamped')
xnew = np.linspace(min(xs) - 4, max(xs) + 4, 101)

splines = [notaknot, natural, clamped]
titles = ['not-a-knot', 'natural', 'clamped']

fig, axs = plt.subplots(3, 3, figsize=(12, 12))
for i in [0, 1, 2]:
    for j, spline, title in zip(range(3), splines, titles):
        axs[i, j].plot(xs, spline(xs, nu=i),'o')
        axs[i, j].plot(xnew, spline(xnew, nu=i),'-')
        axs[i, j].set_title(f'{title}, deriv={i}')

plt.tight_layout()
plt.show()
```

Example 3 (python):
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def add_boundary_knots(spline):
    """
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx,nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])

xs = [1, 2, 3, 4, 5, 6, 7, 8]
ys = [4.5, 3.6, 1.6, 0.0, -3.3, -3.1, -1.8, -1.7]

notaknot = CubicSpline(xs,ys, bc_type='not-a-knot')
# not-a-knot does not require additional intervals

natural = CubicSpline(xs,ys, bc_type='natural')
# extend the natural natural spline with linear extrapolating knots
add_boundary_knots(natural)

clamped = CubicSpline(xs,ys, bc_type='clamped')
# extend the clamped spline with constant extrapolating knots
add_boundary_knots(clamped)

xnew = np.linspace(min(xs) - 5, max(xs) + 5, 201)

fig, axs = plt.subplots(3, 3,figsize=(12,12))

splines = [notaknot, natural, clamped]
titles = ['not-a-knot', 'natural', 'clamped']

for i in [0, 1, 2]:
    for j, spline, title in zip(range(3), splines, titles):
        axs[i, j].plot(xs, spline(xs, nu=i),'o')
        axs[i, j].plot(xnew, spline(xnew, nu=i),'-')
        axs[i, j].set_title(f'{title}, deriv={i}')

plt.tight_layout()
plt.show()
```

Example 4 (python):
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def f(x, a):
    return a*x - 1/np.tan(x)

a = 3
x0 = brentq(f, 1e-16, np.pi/2, args=(a,))   # here we shift the left edge
                                            # by a machine epsilon to avoid
                                            # a division by zero at x=0
xx = np.linspace(0.2, np.pi/2, 101)
plt.plot(xx, a*xx, '--')
plt.plot(xx, 1/np.tan(xx), '--')
plt.plot(x0, a*x0, 'o', ms=12)
plt.text(0.1, 0.9, fr'$x_0 = {x0:.3f}$',
               transform=plt.gca().transAxes, fontsize=16)
plt.show()
```

---

## fft2#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html

**Contents:**
- fft2#

Compute the 2-D discrete Fourier Transform

This function computes the N-D discrete Fourier Transform over any axes in an M-D array by means of the Fast Fourier Transform (FFT). By default, the transform is computed over the last two axes of the input array, i.e., a 2-dimensional FFT.

Input array, can be complex

Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). This corresponds to n for fft(x, n). Along each axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros. if s is not given, the shape of the input along the axes specified by axes is used.

Axes over which to compute the FFT. If not given, the last two axes are used.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axes indicated by axes, or the last two axes if axes is not given.

If s and axes have different length, or axes not given and len(s) != 2.

If an element of axes is larger than the number of axes of x.

Shifts zero-frequency terms to the center of the array. For 2-D input, swaps first and third quadrants, and second and fourth quadrants.

fft2 is just fftn with a different default for axes.

The output, analogously to fft, contains the term for zero frequency in the low-order corner of the transformed axes, the positive frequency terms in the first half of these axes, the term for the Nyquist frequency in the middle of the axes and the negative frequency terms in the second half of the axes, in order of decreasingly negative frequency.

See fftn for details and a plotting example, and fft for definitions and conventions used.

**Examples:**

Example 1 (json):
```json
>>> import scipy.fft
>>> import numpy as np
>>> x = np.mgrid[:5, :5][0]
>>> scipy.fft.fft2(x)
array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary
          0.  +0.j        ,   0.  +0.j        ],
       [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
          0.  +0.j        ,   0.  +0.j        ],
       [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
          0.  +0.j        ,   0.  +0.j        ],
       [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
          0.  +0.j        ,   0.  +0.j        ],
       [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
          0.  +0.j        ,   0.  +0.j        ]])
```

---

## Special Functions (scipy.special)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/special.html

**Contents:**
- Special Functions (scipy.special)#
- Bessel functions of real order(jv, jn_zeros)#
- Cython Bindings for Special Functions (scipy.special.cython_special)#
  - Avoiding Python Function Overhead#
  - Releasing the GIL#
- Functions not in scipy.special#

The main feature of the scipy.special package is the definition of numerous special functions of mathematical physics. Available functions include airy, elliptic, bessel, gamma, beta, hypergeometric, parabolic cylinder, mathieu, spheroidal wave, struve, and kelvin. There are also some low-level stats functions that are not intended for general use as an easier interface to these functions is provided by the stats module. Most of these functions can take array arguments and return array results following the same broadcasting rules as other math functions in Numerical Python. Many of these functions also accept complex numbers as input. For a complete list of the available functions with a one-line description type >>> help(special). Each function also has its own documentation accessible using help. If you don’t see a function you need, consider writing it and contributing it to the library. You can write the function in either C, Fortran, or Python. Look in the source code of the library for examples of each of these kinds of functions.

Bessel functions are a family of solutions to Bessel’s differential equation with real or complex order alpha:

Among other uses, these functions arise in wave propagation problems, such as the vibrational modes of a thin drum head. Here is an example of a circular drum head anchored at the edge:

SciPy also offers Cython bindings for scalar, typed versions of many of the functions in special. The following Cython code gives a simple example of how to use these functions:

(See the Cython documentation for help with compiling Cython.) In the example the function csc.gamma works essentially like its ufunc counterpart gamma, though it takes C types as arguments instead of NumPy arrays. Note, in particular, that the function is overloaded to support real and complex arguments; the correct variant is selected at compile time. The function csc.sici works slightly differently from sici; for the ufunc we could write ai, bi = sici(x), whereas in the Cython version multiple return values are passed as pointers. It might help to think of this as analogous to calling a ufunc with an output array: sici(x, out=(si, ci)).

There are two potential advantages to using the Cython bindings:

they avoid Python function overhead

they do not require the Python Global Interpreter Lock (GIL)

The following sections discuss how to use these advantages to potentially speed up your code, though, of course, one should always profile the code first to make sure putting in the extra effort will be worth it.

For the ufuncs in special, Python function overhead is avoided by vectorizing, that is, by passing an array to the function. Typically, this approach works quite well, but sometimes it is more convenient to call a special function on scalar inputs inside a loop, for example, when implementing your own ufunc. In this case, the Python function overhead can become significant. Consider the following example:

On one computer python_tight_loop took about 131 microseconds to run and cython_tight_loop took about 18.2 microseconds to run. Obviously this example is contrived: one could just call special.jv(np.arange(100), 1) and get results just as fast as in cython_tight_loop. The point is that if Python function overhead becomes significant in your code, then the Cython bindings might be useful.

One often needs to evaluate a special function at many points, and typically the evaluations are trivially parallelizable. Since the Cython bindings do not require the GIL, it is easy to run them in parallel using Cython’s prange function. For example, suppose that we wanted to compute the fundamental solution to the Helmholtz equation:

where \(k\) is the wavenumber and \(\delta\) is the Dirac delta function. It is known that in two dimensions the unique (radiating) solution is

where \(H_0^{(1)}\) is the Hankel function of the first kind, i.e., the function hankel1. The following example shows how we could compute this function in parallel:

(For help with compiling parallel code in Cython see here.) If the above Cython code is in a file test.pyx, then we can write an informal benchmark which compares the parallel and serial versions of the function:

On one quad-core computer the serial method took 1.29 seconds and the parallel method took 0.29 seconds.

Some functions are not included in special because they are straightforward to implement with existing functions in NumPy and SciPy. To prevent reinventing the wheel, this section provides implementations of several such functions, which hopefully illustrate how to handle similar functions. In all examples NumPy is imported as np and special is imported as sc.

The binary entropy function:

A rectangular step function on [0, 1]:

Translating and scaling can be used to get an arbitrary step function.

**Examples:**

Example 1 (python):
```python
>>> from scipy import special
>>> import numpy as np
>>> def drumhead_height(n, k, distance, angle, t):
...    kth_zero = special.jn_zeros(n, k)[-1]
...    return np.cos(t) * np.cos(n*angle) * special.jn(n, distance*kth_zero)
>>> theta = np.r_[0:2*np.pi:50j]
>>> radius = np.r_[0:1:50j]
>>> x = np.array([r * np.cos(theta) for r in radius])
>>> y = np.array([r * np.sin(theta) for r in radius])
>>> z = np.array([drumhead_height(1, 1, r, theta, 0.5) for r in radius])
```

Example 2 (typescript):
```typescript
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> ax = fig.add_axes(rect=(0, 0.05, 0.95, 0.95), projection='3d')
>>> ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
>>> ax.set_xlabel('X')
>>> ax.set_ylabel('Y')
>>> ax.set_xticks(np.arange(-1, 1.1, 0.5))
>>> ax.set_yticks(np.arange(-1, 1.1, 0.5))
>>> ax.set_zlabel('Z')
>>> plt.show()
```

Example 3 (yaml):
```yaml
cimport scipy.special.cython_special as csc

cdef:
    double x = 1
    double complex z = 1 + 1j
    double si, ci, rgam
    double complex cgam

rgam = csc.gamma(x)
print(rgam)
cgam = csc.gamma(z)
print(cgam)
csc.sici(x, &si, &ci)
print(si, ci)
```

Example 4 (python):
```python
import scipy.special as sc
cimport scipy.special.cython_special as csc

def python_tight_loop():
    cdef:
        int n
        double x = 1

    for n in range(100):
        sc.jv(n, x)

def cython_tight_loop():
    cdef:
        int n
        double x = 1

    for n in range(100):
        csc.jv(n, x)
```

---

## Discrete Fourier transforms (scipy.fft)#

**URL:** https://docs.scipy.org/doc/scipy/reference/fft.html

**Contents:**
- Discrete Fourier transforms (scipy.fft)#
- Fast Fourier Transforms (FFTs)#
- Discrete Sin and Cosine Transforms (DST and DCT)#
- Fast Hankel Transforms#
- Helper functions#
- Backend control#

fft(x[, n, axis, norm, overwrite_x, ...])

Compute the 1-D discrete Fourier Transform.

ifft(x[, n, axis, norm, overwrite_x, ...])

Compute the 1-D inverse discrete Fourier Transform.

fft2(x[, s, axes, norm, overwrite_x, ...])

Compute the 2-D discrete Fourier Transform

ifft2(x[, s, axes, norm, overwrite_x, ...])

Compute the 2-D inverse discrete Fourier Transform.

fftn(x[, s, axes, norm, overwrite_x, ...])

Compute the N-D discrete Fourier Transform.

ifftn(x[, s, axes, norm, overwrite_x, ...])

Compute the N-D inverse discrete Fourier Transform.

rfft(x[, n, axis, norm, overwrite_x, ...])

Compute the 1-D discrete Fourier Transform for real input.

irfft(x[, n, axis, norm, overwrite_x, ...])

Computes the inverse of rfft.

rfft2(x[, s, axes, norm, overwrite_x, ...])

Compute the 2-D FFT of a real array.

irfft2(x[, s, axes, norm, overwrite_x, ...])

Computes the inverse of rfft2

rfftn(x[, s, axes, norm, overwrite_x, ...])

Compute the N-D discrete Fourier Transform for real input.

irfftn(x[, s, axes, norm, overwrite_x, ...])

Computes the inverse of rfftn

hfft(x[, n, axis, norm, overwrite_x, ...])

Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum.

ihfft(x[, n, axis, norm, overwrite_x, ...])

Compute the inverse FFT of a signal that has Hermitian symmetry.

hfft2(x[, s, axes, norm, overwrite_x, ...])

Compute the 2-D FFT of a Hermitian complex array.

ihfft2(x[, s, axes, norm, overwrite_x, ...])

Compute the 2-D inverse FFT of a real spectrum.

hfftn(x[, s, axes, norm, overwrite_x, ...])

Compute the N-D FFT of Hermitian symmetric complex input, i.e., a signal with a real spectrum.

ihfftn(x[, s, axes, norm, overwrite_x, ...])

Compute the N-D inverse discrete Fourier Transform for a real spectrum.

dct(x[, type, n, axis, norm, overwrite_x, ...])

Return the Discrete Cosine Transform of arbitrary type sequence x.

idct(x[, type, n, axis, norm, overwrite_x, ...])

Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

dctn(x[, type, s, axes, norm, overwrite_x, ...])

Return multidimensional Discrete Cosine Transform along the specified axes.

idctn(x[, type, s, axes, norm, overwrite_x, ...])

Return multidimensional Inverse Discrete Cosine Transform along the specified axes.

dst(x[, type, n, axis, norm, overwrite_x, ...])

Return the Discrete Sine Transform of arbitrary type sequence x.

idst(x[, type, n, axis, norm, overwrite_x, ...])

Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

dstn(x[, type, s, axes, norm, overwrite_x, ...])

Return multidimensional Discrete Sine Transform along the specified axes.

idstn(x[, type, s, axes, norm, overwrite_x, ...])

Return multidimensional Inverse Discrete Sine Transform along the specified axes.

fht(a, dln, mu[, offset, bias])

Compute the fast Hankel transform.

ifht(A, dln, mu[, offset, bias])

Compute the inverse fast Hankel transform.

Shift the zero-frequency component to the center of the spectrum.

The inverse of fftshift.

fftfreq(n[, d, xp, device])

Return the Discrete Fourier Transform sample frequencies.

rfftfreq(n[, d, xp, device])

Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft).

fhtoffset(dln, mu[, initial, bias])

Return optimal offset for a fast Hankel transform.

next_fast_len(target[, real])

Find the next fast size of input data to fft, for zero-padding, etc.

prev_fast_len(target[, real])

Find the previous fast size of input data to fft.

Context manager for the default number of workers used in scipy.fft

Returns the default number of workers within the current context

set_backend(backend[, coerce, only])

Context manager to set the backend within a fixed scope.

skip_backend(backend)

Context manager to skip a backend within a fixed scope.

set_global_backend(backend[, coerce, only, ...])

Sets the global fft backend

register_backend(backend)

Register a backend for permanent use.

---

## write#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html

**Contents:**
- write#

Write a NumPy array as a WAV file.

The sample rate (in samples/sec).

A 1-D or 2-D NumPy array of either integer or float data-type.

Writes a simple uncompressed WAV file.

To write multiple-channels, use a 2-D array of shape (Nsamples, Nchannels).

The bits-per-sample and PCM/float will be determined by the data-type.

Common data types: [1]

32-bit floating-point

Note that 8-bit PCM is unsigned.

IBM Corporation and Microsoft Corporation, “Multimedia Programming Interface and Data Specifications 1.0”, section “Data Format of the Samples”, August 1991 http://www.tactilemedia.com/info/MCI_Control_Info.html

Create a 100Hz sine wave, sampled at 44100Hz. Write to 16-bit PCM, Mono.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.io.wavfile import write
>>> import numpy as np
>>> samplerate = 44100; fs = 100
>>> t = np.linspace(0., 1., samplerate)
>>> amplitude = np.iinfo(np.int16).max
>>> data = amplitude * np.sin(2. * np.pi * fs * t)
>>> write("example.wav", samplerate, data.astype(np.int16))
```

---

## File IO (scipy.io)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/io.html

**Contents:**
- File IO (scipy.io)#
- MATLAB files#
  - The basic functions#
  - How do I start?#
  - MATLAB structs#
  - MATLAB cell arrays#
- IDL files#
- Matrix Market files#
- Wav sound files (scipy.io.wavfile)#
- Arff files (scipy.io.arff)#

loadmat(file_name[, mdict, appendmat, spmatrix])

savemat(file_name, mdict[, appendmat, ...])

Save a dictionary of names and arrays into a MATLAB-style .mat file.

whosmat(file_name[, appendmat])

List variables inside a MATLAB file.

We’ll start by importing scipy.io and calling it sio for convenience:

If you are using IPython, try tab-completing on sio. Among the many options, you will find:

These are the high-level functions you will most likely use when working with MATLAB files. You’ll also find:

This is the package from which loadmat, savemat, and whosmat are imported. Within sio.matlab, you will find the mio module This module contains the machinery that loadmat and savemat use. From time to time you may find yourself re-using this machinery.

You may have a .mat file that you want to read into SciPy. Or, you want to pass some variables from SciPy / NumPy into MATLAB.

To save us using a MATLAB license, let’s start in Octave. Octave has MATLAB-compatible save and load functions. Start Octave (octave at the command line for me):

Now let’s try the other way round:

If you want to inspect the contents of a MATLAB file without reading the data into memory, use the whosmat command:

whosmat returns a list of tuples, one for each array (or other object) in the file. Each tuple contains the name, shape and data type of the array.

MATLAB structs are a little bit like Python dicts, except the field names must be strings. Any MATLAB object can be a value of a field. As for all objects in MATLAB, structs are, in fact, arrays of structs, where a single struct is an array of shape (1, 1).

We can load this in Python:

In the SciPy versions from 0.12.0, MATLAB structs come back as NumPy structured arrays, with fields named for the struct fields. You can see the field names in the dtype output above. Note also:

So, in MATLAB, the struct array must be at least 2-D, and we replicate that when we read into SciPy. If you want all length 1 dimensions squeezed out, try this:

Sometimes, it’s more convenient to load the MATLAB structs as Python objects rather than NumPy structured arrays - it can make the access syntax in Python a bit more similar to that in MATLAB. In order to do this, use the struct_as_record=False parameter setting to loadmat.

struct_as_record=False works nicely with squeeze_me:

Saving struct arrays can be done in various ways. One simple method is to use dicts:

You can also save structs back again to MATLAB (or Octave in our case) like this:

Cell arrays in MATLAB are rather like Python lists, in the sense that the elements in the arrays can contain any type of MATLAB object. In fact, they are most similar to NumPy object arrays, and that is how we load them into NumPy.

Saving to a MATLAB cell array just involves making a NumPy object array:

readsav(file_name[, idict, python_dict, ...])

Read an IDL .sav file.

Return size and storage parameters from Matrix Market file-like 'source'.

mmread(source, *[, spmatrix])

Reads the contents of a Matrix Market file-like 'source' into a matrix.

mmwrite(target, a[, comment, field, ...])

Writes the sparse or dense array a to Matrix Market file-like target.

read(filename[, mmap])

write(filename, rate, data)

Write a NumPy array as a WAV file.

netcdf_file(filename[, mode, mmap, version, ...])

A file object for NetCDF data.

Allows reading of NetCDF files (version of pupynere package)

**Examples:**

Example 1 (typescript):
```typescript
>>> import scipy.io as sio
```

Example 2 (unknown):
```unknown
sio.loadmat
sio.savemat
sio.whosmat
```

Example 3 (yaml):
```yaml
octave:1> a = 1:12
a =

   1   2   3   4   5   6   7   8   9  10  11  12

octave:2> a = reshape(a, [1 3 4])
a =

ans(:,:,1) =

   1   2   3

ans(:,:,2) =

   4   5   6

ans(:,:,3) =

   7   8   9

ans(:,:,4) =

   10   11   12

octave:3> save -6 octave_a.mat a % MATLAB 6 compatible
octave:4> ls octave_a.mat
octave_a.mat
```

Example 4 (json):
```json
>>> mat_contents = sio.loadmat('octave_a.mat')
>>> mat_contents
{'__header__': b'MATLAB 5.0 MAT-file, written
 by Octave 3.2.3, 2010-05-30 02:13:40 UTC',
 '__version__': '1.0',
 '__globals__': [],
 'a': array([[[ 1.,  4.,  7., 10.],
              [ 2.,  5.,  8., 11.],
              [ 3.,  6.,  9., 12.]]])}
>>> oct_a = mat_contents['a']
>>> oct_a
array([[[  1.,   4.,   7.,  10.],
        [  2.,   5.,   8.,  11.],
        [  3.,   6.,   9.,  12.]]])
>>> oct_a.shape
(1, 3, 4)
```

---

## Finite Difference Differentiation (scipy.differentiate)#

**URL:** https://docs.scipy.org/doc/scipy/reference/differentiate.html

**Contents:**
- Finite Difference Differentiation (scipy.differentiate)#

SciPy differentiate provides functions for performing finite difference numerical differentiation of black-box functions.

derivative(f, x, *[, args, tolerances, ...])

Evaluate the derivative of a elementwise, real scalar function numerically.

jacobian(f, x, *[, tolerances, maxiter, ...])

Evaluate the Jacobian of a function numerically.

hessian(f, x, *[, tolerances, maxiter, ...])

Evaluate the Hessian of a function numerically.

---

## Constants (scipy.constants)#

**URL:** https://docs.scipy.org/doc/scipy/reference/constants.html

**Contents:**
- Constants (scipy.constants)#
- Mathematical constants#
- Physical constants#
  - Constants database#
- Units#
  - SI prefixes#
  - Binary prefixes#
  - Mass#
  - Angle#
  - Time#

Physical and mathematical constants and units.

The following physical constants are available as attributes of scipy.constants. All units are SI.

speed of light in vacuum

speed of light in vacuum

the magnetic constant \(\mu_0\)

the electric constant (vacuum permittivity), \(\epsilon_0\)

the Planck constant \(h\)

the Planck constant \(h\)

the reduced Planck constant, \(\hbar = h/(2\pi)\)

Newtonian constant of gravitation

gravitational_constant

Newtonian constant of gravitation

standard acceleration of gravity

fine-structure constant

fine-structure constant

Stefan-Boltzmann constant \(\sigma\)

Stefan-Boltzmann constant \(\sigma\)

Wien wavelength displacement law constant

In addition to the above variables, scipy.constants also contains the 2022 CODATA recommended values [CODATA2022] database containing more physical constants.

Value in physical_constants indexed by key

Unit in physical_constants indexed by key

Relative precision in physical_constants indexed by key

Return list of physical_constant keys containing a given string.

Accessing a constant no longer in current CODATA data set

Dictionary of physical constants, of the format physical_constants[name] = (value, unit, uncertainty). The CODATA database uses ellipses to indicate that a value is defined (exactly) in terms of others but cannot be represented exactly with the allocated number of digits. In these cases, SciPy calculates the derived value and reports it to the full precision of a Python float. Although physical_constants lists the uncertainty as 0.0 to indicate that the CODATA value is exact, the value in physical_constants is still subject to the truncation error inherent in double-precision representation.

alpha particle mass energy equivalent

alpha particle mass energy equivalent in MeV

alpha particle mass in u

alpha particle molar mass

0.0040015061833 kg mol^-1

alpha particle relative atomic mass

alpha particle rms charge radius

alpha particle-electron mass ratio

alpha particle-proton mass ratio

atomic mass constant energy equivalent

atomic mass constant energy equivalent in MeV

atomic mass unit-electron volt relationship

atomic mass unit-hartree relationship

atomic mass unit-hertz relationship

atomic mass unit-inverse meter relationship

751300662090000.0 m^-1

atomic mass unit-joule relationship

atomic mass unit-kelvin relationship

atomic mass unit-kilogram relationship

atomic unit of 1st hyperpolarizability

3.2063612996e-53 C^3 m^3 J^-2

atomic unit of 2nd hyperpolarizability

6.2353799735e-65 C^4 m^4 J^-3

atomic unit of action

1.0545718176461565e-34 J s

atomic unit of charge

atomic unit of charge density

1081202386770.0 C m^-3

atomic unit of current

atomic unit of electric dipole mom.

atomic unit of electric field

514220675112.0 V m^-1

atomic unit of electric field gradient

9.7173624424e+21 V m^-2

atomic unit of electric polarizability

1.64877727212e-41 C^2 m^2 J^-1

atomic unit of electric potential

atomic unit of electric quadrupole mom.

4.4865515185e-40 C m^2

atomic unit of energy

atomic unit of length

atomic unit of mag. dipole mom.

1.85480201315e-23 J T^-1

atomic unit of mag. flux density

atomic unit of magnetizability

7.8910365794e-29 J T^-2

atomic unit of momentum

1.99285191545e-24 kg m s^-1

atomic unit of permittivity

1.1126500562e-10 F m^-1

2.4188843265864e-17 s

atomic unit of velocity

6.02214076e+23 mol^-1

9.2740100657e-24 J T^-1

Bohr magneton in eV/T

5.7883817982e-05 eV T^-1

Bohr magneton in Hz/T

13996244917.1 Hz T^-1

Bohr magneton in inverse meter per tesla

46.686447719 m^-1 T^-1

Boltzmann constant in eV/K

8.617333262145179e-05 eV K^-1

Boltzmann constant in Hz/K

20836619123.327576 Hz K^-1

Boltzmann constant in inverse meter per kelvin

69.50348004861274 m^-1 K^-1

characteristic impedance of vacuum

classical electron radius

7.748091729863649e-05 S

conventional value of ampere-90

conventional value of coulomb-90

conventional value of farad-90

conventional value of henry-90

conventional value of Josephson constant

483597900000000.0 Hz V^-1

conventional value of ohm-90

1.0000000177936679 ohm

conventional value of volt-90

conventional value of von Klitzing constant

conventional value of watt-90

4.330735087e-27 J T^-1

deuteron mag. mom. to Bohr magneton ratio

deuteron mag. mom. to nuclear magneton ratio

deuteron mass energy equivalent

deuteron mass energy equivalent in MeV

0.00201355321466 kg mol^-1

deuteron relative atomic mass

deuteron rms charge radius

deuteron-electron mag. mom. ratio

deuteron-electron mass ratio

deuteron-neutron mag. mom. ratio

deuteron-proton mag. mom. ratio

deuteron-proton mass ratio

electron charge to mass quotient

-175882000838.0 C kg^-1

electron gyromag. ratio

176085962784.0 s^-1 T^-1

electron gyromag. ratio in MHz/T

28024.9513861 MHz T^-1

-9.2847646917e-24 J T^-1

electron mag. mom. anomaly

electron mag. mom. to Bohr magneton ratio

electron mag. mom. to nuclear magneton ratio

electron mass energy equivalent

electron mass energy equivalent in MeV

5.4857990962e-07 kg mol^-1

electron relative atomic mass

electron to alpha particle mass ratio

electron to shielded helion mag. mom. ratio

electron to shielded proton mag. mom. ratio

electron volt-atomic mass unit relationship

electron volt-hartree relationship

0.036749322175665 E_h

electron volt-hertz relationship

electron volt-inverse meter relationship

806554.3937349211 m^-1

electron volt-joule relationship

electron volt-kelvin relationship

electron volt-kilogram relationship

1.7826619216278975e-36 kg

electron-deuteron mag. mom. ratio

electron-deuteron mass ratio

electron-helion mass ratio

electron-muon mag. mom. ratio

electron-muon mass ratio

electron-neutron mag. mom. ratio

electron-neutron mass ratio

electron-proton mag. mom. ratio

electron-proton mass ratio

electron-tau mass ratio

electron-triton mass ratio

elementary charge over h-bar

1519267447878626.0 A J^-1

96485.33212331001 C mol^-1

Fermi coupling constant

fine-structure constant

first radiation constant

3.7417718521927573e-16 W m^2

first radiation constant for spectral radiance

1.1910429723971884e-16 W m^2 sr^-1

hartree-atomic mass unit relationship

hartree-electron volt relationship

hartree-hertz relationship

6579683920499900.0 Hz

hartree-inverse meter relationship

hartree-joule relationship

hartree-kelvin relationship

hartree-kilogram relationship

4.8508702095419e-35 kg

-1.07461755198e-26 J T^-1

helion mag. mom. to Bohr magneton ratio

helion mag. mom. to nuclear magneton ratio

helion mass energy equivalent

helion mass energy equivalent in MeV

0.0030149322501 kg mol^-1

helion relative atomic mass

helion shielding shift

helion-electron mass ratio

helion-proton mass ratio

hertz-atomic mass unit relationship

hertz-electron volt relationship

4.135667696923859e-15 eV

hertz-hartree relationship

1.5198298460574e-16 E_h

hertz-inverse meter relationship

3.3356409519815204e-09 m^-1

hertz-joule relationship

hertz-kelvin relationship

4.799243073366221e-11 K

hertz-kilogram relationship

7.372497323812708e-51 kg

hyperfine transition frequency of Cs-133

inverse fine-structure constant

inverse meter-atomic mass unit relationship

inverse meter-electron volt relationship

1.2398419843320026e-06 eV

inverse meter-hartree relationship

4.5563352529132e-08 E_h

inverse meter-hertz relationship

inverse meter-joule relationship

1.9864458571489286e-25 J

inverse meter-kelvin relationship

0.014387768775039337 K

inverse meter-kilogram relationship

2.2102190943042335e-42 kg

inverse of conductance quantum

12906.403729652257 ohm

483597848416983.6 Hz V^-1

joule-atomic mass unit relationship

joule-electron volt relationship

6.241509074460763e+18 eV

joule-hartree relationship

2.2937122783969e+17 E_h

joule-hertz relationship

1.5091901796421518e+33 Hz

joule-inverse meter relationship

5.03411656754271e+24 m^-1

joule-kelvin relationship

7.24297051603992e+22 K

joule-kilogram relationship

1.1126500560536185e-17 kg

kelvin-atomic mass unit relationship

kelvin-electron volt relationship

8.617333262145179e-05 eV

kelvin-hartree relationship

3.1668115634564e-06 E_h

kelvin-hertz relationship

20836619123.327576 Hz

kelvin-inverse meter relationship

69.50348004861274 m^-1

kelvin-joule relationship

kelvin-kilogram relationship

1.5361791872403723e-40 kg

kilogram-atomic mass unit relationship

kilogram-electron volt relationship

5.609588603804452e+35 eV

kilogram-hartree relationship

2.0614857887415e+34 E_h

kilogram-hertz relationship

1.3563924896521321e+50 Hz

kilogram-inverse meter relationship

4.524438335443823e+41 m^-1

kilogram-joule relationship

8.987551787368176e+16 J

kilogram-kelvin relationship

6.509657260728958e+39 K

lattice parameter of silicon

lattice spacing of ideal Si (220)

Loschmidt constant (273.15 K, 100 kPa)

2.6516458048837345e+25 m^-3

Loschmidt constant (273.15 K, 101.325 kPa)

2.686780111798444e+25 m^-3

2.0678338484619295e-15 Wb

8.31446261815324 J mol^-1 K^-1

0.00100000000105 kg mol^-1

molar mass of carbon-12

0.0120000000126 kg mol^-1

molar Planck constant

3.990312712893431e-10 J Hz^-1 mol^-1

molar volume of ideal gas (273.15 K, 100 kPa)

0.02271095464148557 m^3 mol^-1

molar volume of ideal gas (273.15 K, 101.325 kPa)

0.022413969545014137 m^3 mol^-1

molar volume of silicon

1.205883199e-05 m^3 mol^-1

muon Compton wavelength

-4.4904483e-26 J T^-1

muon mag. mom. anomaly

muon mag. mom. to Bohr magneton ratio

muon mag. mom. to nuclear magneton ratio

muon mass energy equivalent

muon mass energy equivalent in MeV

0.0001134289258 kg mol^-1

muon-electron mass ratio

muon-neutron mass ratio

muon-proton mag. mom. ratio

muon-proton mass ratio

natural unit of action

1.0545718176461565e-34 J s

natural unit of action in eV s

6.582119569509067e-16 eV s

natural unit of energy

natural unit of energy in MeV

natural unit of length

natural unit of momentum

2.730924488e-22 kg m s^-1

natural unit of momentum in MeV/c

natural unit of velocity

neutron Compton wavelength

neutron gyromag. ratio

183247174.0 s^-1 T^-1

neutron gyromag. ratio in MHz/T

-9.6623653e-27 J T^-1

neutron mag. mom. to Bohr magneton ratio

neutron mag. mom. to nuclear magneton ratio

neutron mass energy equivalent

neutron mass energy equivalent in MeV

0.00100866491712 kg mol^-1

neutron relative atomic mass

neutron to shielded proton mag. mom. ratio

neutron-electron mag. mom. ratio

neutron-electron mass ratio

neutron-muon mass ratio

neutron-proton mag. mom. ratio

neutron-proton mass difference

neutron-proton mass difference energy equivalent

neutron-proton mass difference energy equivalent in MeV

neutron-proton mass difference in u

neutron-proton mass ratio

neutron-tau mass ratio

Newtonian constant of gravitation

6.6743e-11 m^3 kg^-1 s^-2

Newtonian constant of gravitation over h-bar c

6.70883e-39 (GeV/c^2)^-2

5.0507837393e-27 J T^-1

nuclear magneton in eV/T

3.15245125417e-08 eV T^-1

nuclear magneton in inverse meter per tesla

0.0254262341009 m^-1 T^-1

nuclear magneton in K/T

0.00036582677706 K T^-1

nuclear magneton in MHz/T

7.6225932188 MHz T^-1

6.62607015e-34 J Hz^-1

Planck constant in eV/Hz

4.135667696923859e-15 eV Hz^-1

Planck mass energy equivalent in GeV

proton charge to mass quotient

proton Compton wavelength

proton gyromag. ratio

267522187.08 s^-1 T^-1

proton gyromag. ratio in MHz/T

42.577478461 MHz T^-1

1.41060679545e-26 J T^-1

proton mag. mom. to Bohr magneton ratio

proton mag. mom. to nuclear magneton ratio

proton mag. shielding correction

proton mass energy equivalent

proton mass energy equivalent in MeV

0.00100727646764 kg mol^-1

proton relative atomic mass

proton rms charge radius

proton-electron mass ratio

proton-muon mass ratio

proton-neutron mag. mom. ratio

proton-neutron mass ratio

proton-tau mass ratio

quantum of circulation

0.00036369475467 m^2 s^-1

quantum of circulation times 2

0.00072738950934 m^2 s^-1

reduced Compton wavelength

reduced muon Compton wavelength

reduced neutron Compton wavelength

reduced Planck constant

1.0545718176461565e-34 J s

reduced Planck constant in eV s

6.582119569509067e-16 eV s

reduced Planck constant times c in MeV fm

197.3269804593025 MeV fm

reduced proton Compton wavelength

reduced tau Compton wavelength

Rydberg constant times c in Hz

3289841960250000.0 Hz

Rydberg constant times hc in eV

Rydberg constant times hc in J

Sackur-Tetrode constant (1 K, 100 kPa)

Sackur-Tetrode constant (1 K, 101.325 kPa)

second radiation constant

0.014387768775039337 m K

shielded helion gyromag. ratio

203789460.78 s^-1 T^-1

shielded helion gyromag. ratio in MHz/T

32.434100033 MHz T^-1

shielded helion mag. mom.

-1.07455311035e-26 J T^-1

shielded helion mag. mom. to Bohr magneton ratio

shielded helion mag. mom. to nuclear magneton ratio

shielded helion to proton mag. mom. ratio

shielded helion to shielded proton mag. mom. ratio

shielded proton gyromag. ratio

267515319.4 s^-1 T^-1

shielded proton gyromag. ratio in MHz/T

shielded proton mag. mom.

1.410570583e-26 J T^-1

shielded proton mag. mom. to Bohr magneton ratio

shielded proton mag. mom. to nuclear magneton ratio

shielding difference of d and p in HD

shielding difference of t and p in HT

speed of light in vacuum

standard acceleration of gravity

standard-state pressure

Stefan-Boltzmann constant

5.6703744191844314e-08 W m^-2 K^-4

tau Compton wavelength

tau energy equivalent

tau mass energy equivalent

tau-electron mass ratio

tau-neutron mass ratio

tau-proton mass ratio

Thomson cross section

1.5046095178e-26 J T^-1

triton mag. mom. to Bohr magneton ratio

triton mag. mom. to nuclear magneton ratio

triton mass energy equivalent

triton mass energy equivalent in MeV

0.00301550071913 kg mol^-1

triton relative atomic mass

triton to proton mag. mom. ratio

triton-electron mass ratio

triton-proton mass ratio

unified atomic mass unit

vacuum electric permittivity

8.8541878188e-12 F m^-1

vacuum mag. permeability

1.25663706127e-06 N A^-2

von Klitzing constant

25812.807459304513 ohm

Wien frequency displacement law constant

58789257576.468254 Hz K^-1

Wien wavelength displacement law constant

0.0028977719551851727 m K

one pound (avoirdupous) in kg

one pound (avoirdupous) in kg

one inch version of a slug in kg (added in 1.0.0)

one inch version of a slug in kg (added in 1.0.0)

one slug in kg (added in 1.0.0)

atomic mass constant (in kg)

atomic mass constant (in kg)

atomic mass constant (in kg)

arc minute in radians

arc minute in radians

arc second in radians

arc second in radians

one minute in seconds

one year (365 days) in seconds

one Julian year (365.25 days) in seconds

one survey foot in meters

one survey mile in meters

one nautical mile in meters

one Angstrom in meters

one astronomical unit in meters

one astronomical unit in meters

one light year in meters

standard atmosphere in pascals

standard atmosphere in pascals

one torr (mmHg) in pascals

one torr (mmHg) in pascals

one hectare in square meters

one acre in square meters

one liter in cubic meters

one liter in cubic meters

one gallon (US) in cubic meters

one gallon (US) in cubic meters

one gallon (UK) in cubic meters

one fluid ounce (US) in cubic meters

one fluid ounce (US) in cubic meters

one fluid ounce (UK) in cubic meters

one barrel in cubic meters

one barrel in cubic meters

kilometers per hour in meters per second

miles per hour in meters per second

one Mach (approx., at 15 C, 1 atm) in meters per second

one Mach (approx., at 15 C, 1 atm) in meters per second

one knot in meters per second

zero of Celsius scale in Kelvin

one Fahrenheit (only differences) in Kelvins

convert_temperature(val, old_scale, new_scale)

Convert from a temperature scale to another one among Celsius, Kelvin, Fahrenheit, and Rankine scales.

one electron volt in Joules

one electron volt in Joules

one calorie (thermochemical) in Joules

one calorie (thermochemical) in Joules

one calorie (International Steam Table calorie, 1956) in Joules

one British thermal unit (International Steam Table) in Joules

one British thermal unit (International Steam Table) in Joules

one British thermal unit (thermochemical) in Joules

one ton of TNT in Joules

one horsepower in watts

one horsepower in watts

one pound force in newtons

one pound force in newtons

one kilogram force in newtons

one kilogram force in newtons

Convert wavelength to optical frequency

Convert optical frequency to wavelength.

CODATA Recommended Values of the Fundamental Physical Constants 2022.

https://physics.nist.gov/cuu/Constants/

---

## 1-D interpolation#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html

**Contents:**
- 1-D interpolation#
- Piecewise linear interpolation#
- Cubic splines#
- Monotone interpolants#
- Interpolation with B-splines#
  - Non-cubic splines#
- Batches of y#
- Parametric spline curves#
- Missing data#
- Legacy interface for 1-D interpolation (interp1d)#

If all you need is a linear (a.k.a. broken line) interpolation, you can use the numpy.interp routine. It takes two arrays of data to interpolate, x, and y, and a third array, xnew, of points to evaluate the interpolation on:

Construct the interpolation

One limitation of numpy.interp is that it does not allow controlling the extrapolation. See the interpolation with B-Splines section section for alternative routines which provide this kind of functionality.

Of course, piecewise linear interpolation produces corners at data points, where linear pieces join. To produce a smoother curve, you can use cubic splines, where the interpolating curve is made of cubic pieces with matching first and second derivatives. In code, these objects are represented via the CubicSpline class instances. An instance is constructed with the x and y arrays of data, and then it can be evaluated using the target xnew values:

A CubicSpline object’s __call__ method accepts both scalar values and arrays. It also accepts a second argument, nu, to evaluate the derivative of order nu. As an example, we plot the derivatives of a spline:

Note that the first and second derivatives are continuous by construction, and the third derivative jumps at data points.

Cubic splines are by construction twice continuously differentiable. This may lead to the spline function oscillating and ‘’overshooting’’ in between the data points. In these situations, an alternative is to use the so-called monotone cubic interpolants: these are constructed to be only once continuously differentiable, and attempt to preserve the local shape implied by the data. scipy.interpolate provides two objects of this kind: PchipInterpolator and Akima1DInterpolator . To illustrate, let’s consider data with an outlier:

B-splines form an alternative (if formally equivalent) representation of piecewise polynomials. This basis is generally more computationally stable than the power basis and is useful for a variety of applications which include interpolation, regression and curve representation. Details are given in the piecewise polynomials section, and here we illustrate their usage by constructing the interpolation of a sine function:

To construct the interpolating objects given data arrays, x and y, we use the make_interp_spline function:

This function returns an object which has an interface similar to that of the CubicSpline objects. In particular, it can be evaluated at a data point and differentiated:

Note that by specifying k=3 in the make_interp_spline call, we requested a cubic spline (this is the default, so k=3 could have been omitted); the derivative of a cubic is a quadratic:

By default, the result of make_interp_spline(x, y) is equivalent to CubicSpline(x, y). The difference is that the former allows several optional capabilities: it can construct splines of various degrees (via the optional argument k) and predefined knots (via the optional argument t).

Boundary conditions for the spline interpolation can be controlled by the bc_type argument to make_interp_spline function and CubicSpline constructor. By default, both use the ‘not-a-knot’ boundary condition.

One use of make_interp_spline is constructing a linear interpolant with linear extrapolation since make_interp_spline extrapolates by default. Consider

See the extrapolation section for more details and discussion.

Univariate interpolators accept not only one-dimensional y arrays, but also y.ndim > 1. The interpretation is that y is a batch of 1D data arrays: by default, the zeroth dimension of y is the interpolation axis, and the trailing dimensions are batch dimensions. Consider a collection (a batch) of functions \(f_j\) sampled at the points \(x_i\). We can instantiate a single interpolator for all of these functions by providing a two-dimensional array y such that y[i, j] records \(f_j(x_i)\).

Several notes are in order. First and foremost, the behavior here looks similar to NumPy’s broadcasting, but differs in two respects:

The x array is expected to be 1D even if the y array is not: x.ndim == 1 while y.ndim >= 1. There is no broadcasting of x vs y.

By default, the trailing dimensions are used as batch dimensions, in contrast to the NumPy convention of using the leading dimensions as batch dimensions.

Second, the interpolation axis can be controlled by an optional axis argument. The example above uses the default value of axis=0. For a non-default values, the following is true:

y.shape[axis] == x.size (otherwise en error is raised)

the shape of spl(xv) is y.shape[axis:] + xv.shape + y.shape[:axis]

While we demonstrated the batching behavior with make_interp_spline, in fact the majority of univariate interpolators support this functionality: PchipInterpolator and Akima1DInterpolator, CubicSpline; low-level polynomial representation classes, PPoly, BPoly and BSpline; as well as least-squares fit and spline smoothing functions, make_lsq_spline and make_smoothing_spline.

So far we considered spline functions, where the data, y, is expected to depend explicitly on the independent variable x—so that the interpolating function satisfies \(f(x_j) = y_j\). Spline curves treat the x and y arrays as coordinates of points, \(\mathbf{p}_j\) on a plane, and an interpolating curve which passes through these points is parameterized by some additional parameter (typically called u). Note that this construction readily generalizes to higher dimensions where \(\mathbf{p}_j\) are points in an N-dimensional space.

Spline curves can be easily constructed using the fact that interpolation functions handle multidimensional data arrays, as discussed in the previous section. The values of the parameter, u, corresponding to the data points, need to be separately supplied by the user.

The choice of parametrization is problem-dependent and different parametrizations may produce vastly different curves. As an example, we consider three parametrizations of (a somewhat difficult) dataset, which we take from Chapter 6 of Ref [1] listed in the BSpline docstring:

We take elements of the p array as coordinates of seven points on the plane, where p[:, j] gives the coordinates of the point \(\mathbf{p}_j\).

First, consider the uniform parametrization, \(u_j = j\):

Second, we consider the so-called cord length parametrization, which is nothing but a cumulative length of straight line segments connecting the data points:

for \(j=1, 2, \dots\) and \(u_0 = 0\). Here \(| \cdots |\) is the length between the consecutive points \(p_j\) on the plane.

Finally, we consider what is sometimes called the centripetal parametrization: \(u_j = u_{j-1} + |\mathbf{p}_j - \mathbf{p}_{j-1}|^{1/2}\). Due to the extra square root, the difference between consecutive values \(u_j - u_{j-1}\) will be smaller than for the cord length parametrization:

Now plot the resulting curves:

We note that scipy.interpolate does not support interpolation with missing data. Two popular ways of representing missing data are using masked arrays of the numpy.ma library, and encoding missing values as not-a-number, NaN.

Neither of these two approaches is directly supported in scipy.interpolate. Individual routines may offer partial support, and/or workarounds, but in general, the library firmly adheres to the IEEE 754 semantics where a NaN means not-a-number, i.e. a result of an illegal mathematical operation (e.g., division by zero), not missing.

interp1d is considered legacy API and is not recommended for use in new code. Consider using more specific interpolators instead.

The interp1d class in scipy.interpolate is a convenient method to create a function based on fixed data points, which can be evaluated anywhere within the domain defined by the given data using linear interpolation. An instance of this class is created by passing the 1-D vectors comprising the data. The instance of this class defines a __call__ method and can therefore be treated like a function which interpolates between known data values to obtain unknown values. Behavior at the boundary can be specified at instantiation time. The following example demonstrates its use, for linear and cubic spline interpolation:

The ‘cubic’ kind of interp1d is equivalent to make_interp_spline, and the ‘linear’ kind is equivalent to numpy.interp while also allowing N-dimensional y arrays.

Another set of interpolations in interp1d is nearest, previous, and next, where they return the nearest, previous, or next point along the x-axis. Nearest and next can be thought of as a special case of a causal interpolating filter. The following example demonstrates their use, using the same data as in the previous example:

As mentioned, interp1d class is legacy: we have no plans to remove it; we are going to keep supporting its existing usages; however we believe there are better alternatives which we recommend using in new code.

Here we list specific recommendations, depending on the interpolation kind.

Linear interpolation, kind="linear"

The default recommendation is to use numpy.interp function. Alternatively, you can use linear splines, make_interp_spline(x, y, k=1), see this section for a discussion.

Spline interpolators, kind="quadratic" or "cubic"

Under the hood, interp1d delegates to make_interp_spline, so we recommend using the latter directly.

Piecewise constant modes, kind="nearest", "previous", "next"

First, we note that interp1d(x, y, kind='previous') is equivalent to make_interp_spline(x, y, k=0).

More generally however, all these piecewise constant interpolation modes are based on numpy.searchsorted. For example, the "nearest" mode is nothing but

Other variants are similar, see the interp1d source code for details.

**Examples:**

Example 1 (typescript):
```typescript
>>> import numpy as np
>>> x = np.linspace(0, 10, num=11)
>>> y = np.cos(-x**2 / 9.0)
```

Example 2 (unknown):
```unknown
>>> xnew = np.linspace(0, 10, num=1001)
>>> ynew = np.interp(xnew, x, y)
```

Example 3 (typescript):
```typescript
>>> import matplotlib.pyplot as plt
>>> plt.plot(xnew, ynew, '-', label='linear interp')
>>> plt.plot(x, y, 'o', label='data')
>>> plt.legend(loc='best')
>>> plt.show()
```

Example 4 (sql):
```sql
>>> from scipy.interpolate import CubicSpline
>>> spl = CubicSpline([1, 2, 3, 4, 5, 6], [1, 4, 8, 16, 25, 36])
>>> spl(2.5)
5.57
```

---

## tanhm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.tanhm.html

**Contents:**
- tanhm#

Compute the hyperbolic matrix tangent.

This routine uses expm to compute the matrix exponentials.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Hyperbolic matrix tangent of A

Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import tanhm, sinhm, coshm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> t = tanhm(a)
>>> t
array([[ 0.3428582 ,  0.51987926],
       [ 0.17329309,  0.86273746]])
```

Example 2 (json):
```json
>>> s = sinhm(a)
>>> c = coshm(a)
>>> t - s.dot(np.linalg.inv(c))
array([[  2.72004641e-15,   4.55191440e-15],
       [  0.00000000e+00,  -5.55111512e-16]])
```

---

## Statistical functions for masked arrays (scipy.stats.mstats)#

**URL:** https://docs.scipy.org/doc/scipy/reference/stats.mstats.html

**Contents:**
- Statistical functions for masked arrays (scipy.stats.mstats)#
- Summary statistics#
- Frequency statistics#
- Correlation functions#
- Statistical tests#
- Transformations#
- Other#

This module contains a large number of statistical functions that can be used with masked arrays.

Most of these functions are similar to those in scipy.stats but might have small differences in the API or in the algorithm used. Since this is a relatively new package, some API changes are still possible.

describe(a[, axis, ddof, bias])

Computes several descriptive statistics of the passed array.

gmean(a[, axis, dtype, weights, nan_policy, ...])

Compute the weighted geometric mean along the specified axis.

hmean(a[, axis, dtype, weights, nan_policy, ...])

Calculate the weighted harmonic mean along the specified axis.

kurtosis(a[, axis, fisher, bias])

Computes the kurtosis (Fisher or Pearson) of a dataset.

Returns an array of the modal (most common) value in the passed array.

mquantiles(a[, prob, alphap, betap, axis, limit])

Computes empirical quantiles for a data array.

hdmedian(data[, axis, var])

Returns the Harrell-Davis estimate of the median along the given axis.

hdquantiles(data[, prob, axis, var])

Computes quantile estimates with the Harrell-Davis method.

hdquantiles_sd(data[, prob, axis])

The standard error of the Harrell-Davis quantile estimates by jackknife.

idealfourths(data[, axis])

Returns an estimate of the lower and upper quartiles.

plotting_positions(data[, alpha, beta])

Returns plotting positions (or empirical percentile points) for the data.

meppf(data[, alpha, beta])

Returns plotting positions (or empirical percentile points) for the data.

moment(a[, moment, axis])

Calculates the nth moment about the mean for a sample.

skew(a[, axis, bias])

Computes the skewness of a data set.

tmean(a[, limits, inclusive, axis])

Compute the trimmed mean.

tvar(a[, limits, inclusive, axis, ddof])

Compute the trimmed variance

tmin(a[, lowerlimit, axis, inclusive])

Compute the trimmed minimum

tmax(a[, upperlimit, axis, inclusive])

Compute the trimmed maximum

tsem(a[, limits, inclusive, axis, ddof])

Compute the trimmed standard error of the mean.

variation(a[, axis, ddof])

Compute the coefficient of variation.

Find repeats in arr and return a tuple (repeats, repeat_count).

Calculates the standard error of the mean of the input array.

trimmed_mean(a[, limits, inclusive, ...])

Returns the trimmed mean of the data along the given axis.

trimmed_mean_ci(data[, limits, inclusive, ...])

Selected confidence interval of the trimmed mean along the given axis.

trimmed_std(a[, limits, inclusive, ...])

Returns the trimmed standard deviation of the data along the given axis.

trimmed_var(a[, limits, inclusive, ...])

Returns the trimmed variance of the data along the given axis.

scoreatpercentile(data, per[, limit, ...])

Calculate the score at the given 'per' percentile of the sequence a.

Performs a 1-way ANOVA, returning an F-value and probability given any number of groups.

Pearson correlation coefficient and p-value for testing non-correlation.

spearmanr(x[, y, use_ties, axis, ...])

Calculates a Spearman rank-order correlation coefficient and the p-value to test for non-correlation.

Calculates a point biserial correlation coefficient and its p-value.

kendalltau(x, y[, use_ties, use_missing, ...])

Computes Kendall's rank correlation tau on two variables x and y.

kendalltau_seasonal(x)

Computes a multivariate Kendall's rank correlation tau, for seasonal data.

Calculate a linear least-squares regression for two sets of measurements.

siegelslopes(y[, x, method])

Computes the Siegel estimator for a set of points (x, y).

theilslopes(y[, x, alpha, method])

Computes the Theil-Sen estimator for a set of points (x, y).

sen_seasonal_slopes(x)

Computes seasonal Theil-Sen and Kendall slope estimators.

ttest_1samp(a, popmean[, axis, alternative])

Calculates the T-test for the mean of ONE group of scores.

ttest_onesamp(a, popmean[, axis, alternative])

Calculates the T-test for the mean of ONE group of scores.

ttest_ind(a, b[, axis, equal_var, alternative])

Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

ttest_rel(a, b[, axis, alternative])

Calculates the T-test on TWO RELATED samples of scores, a and b.

chisquare(f_obs[, f_exp, ddof, axis, ...])

Perform Pearson's chi-squared test.

kstest(data1, data2[, args, alternative, method])

ks_2samp(data1, data2[, alternative, method])

Computes the Kolmogorov-Smirnov test on two samples.

ks_1samp(x, cdf[, args, alternative, method])

Computes the Kolmogorov-Smirnov test on one sample of masked values.

ks_twosamp(data1, data2[, alternative, method])

Computes the Kolmogorov-Smirnov test on two samples.

mannwhitneyu(x, y[, use_continuity])

Computes the Mann-Whitney statistic

rankdata(data[, axis, use_missing])

Returns the rank (also known as order statistics) of each data point along the given axis.

Compute the Kruskal-Wallis H-test for independent samples

Compute the Kruskal-Wallis H-test for independent samples

friedmanchisquare(*args)

Friedman Chi-Square is a non-parametric, one-way within-subjects ANOVA.

brunnermunzel(x, y[, alternative, distribution])

Compute the Brunner-Munzel test on samples x and y.

skewtest(a[, axis, alternative])

Tests whether the skew is different from the normal distribution.

kurtosistest(a[, axis, alternative])

Tests whether a dataset has normal kurtosis

normaltest(a[, axis])

Tests whether a sample differs from a normal distribution.

obrientransform(*args)

Computes a transform on input data (any number of columns).

trim(a[, limits, inclusive, relative, axis])

Trims an array by masking the data outside some given limits.

trima(a[, limits, inclusive])

Trims an array by masking the data outside some given limits.

trimmed_stde(a[, limits, inclusive, axis])

Returns the standard error of the trimmed mean along the given axis.

trimr(a[, limits, inclusive, axis])

Trims an array by masking some proportion of the data on each end.

trimtail(data[, proportiontocut, tail, ...])

Trims the data by masking values from one tail.

trimboth(data[, proportiontocut, inclusive, ...])

Trims the smallest and largest data values.

winsorize(a[, limits, inclusive, inplace, ...])

Returns a Winsorized version of the input array.

zmap(scores, compare[, axis, ddof, nan_policy])

Calculate the relative z-scores.

zscore(a[, axis, ddof, nan_policy])

Constructs a 2D array from a group of sequences.

count_tied_groups(x[, use_missing])

Counts the number of tied values.

Returns the sign of x, or 0 if x is masked.

compare_medians_ms(group_1, group_2[, axis])

Compares the medians from two independent groups along the given axis.

median_cihs(data[, alpha, axis])

Computes the alpha-level confidence interval for the median of the data.

mjci(data[, prob, axis])

Returns the Maritz-Jarrett estimators of the standard error of selected experimental quantiles of the data.

mquantiles_cimj(data[, prob, alpha, axis])

Computes the alpha confidence interval for the selected quantiles of the data, with Maritz-Jarrett estimators.

Evaluates Rosenblatt's shifted histogram estimators for each data point.

---

## make_lsq_spline#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_lsq_spline.html

**Contents:**
- make_lsq_spline#

Create a smoothing B-spline satisfying the Least SQuares (LSQ) criterion.

The result is a linear combination

of the B-spline basis elements, \(B_j(x; t)\), which minimizes

Knots. Knots and data points must satisfy Schoenberg-Whitney conditions.

B-spline degree. Default is cubic, k = 3.

Weights for spline fitting. Must be positive. If None, then weights are all equal. Default is None.

Interpolation axis. Default is zero.

Whether to check that the input arrays contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs. Default is True.

Method for solving the linear LSQ problem. Allowed values are “norm-eq” (Explicitly construct and solve the normal system of equations), and “qr” (Use the QR factorization of the design matrix). Default is “qr”.

A BSpline object of the degree k with knots t.

base class representing the B-spline objects

a similar factory function for interpolating splines

a FITPACK-based spline fitting routine

a FITPACK-based fitting routine

The number of data points must be larger than the spline degree k.

Knots t must satisfy the Schoenberg-Whitney conditions, i.e., there must be a subset of data points x[j] such that t[j] < x[j] < t[j+k+1], for j=0, 1,...,n-k-2.

Generate some noisy data:

Now fit a smoothing cubic spline with a pre-defined internal knots. Here we make the knot vector (k+1)-regular by adding boundary knots:

For comparison, we also construct an interpolating spline for the same set of data:

NaN handling: If the input arrays contain nan values, the result is not useful since the underlying spline fitting routines cannot deal with nan. A workaround is to use zero weights for not-a-number data points:

Notice the need to replace a nan by a numerical value (precise value does not matter as long as the corresponding weight is zero.)

**Examples:**

Example 1 (typescript):
```typescript
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> x = np.linspace(-3, 3, 50)
>>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
```

Example 2 (sql):
```sql
>>> from scipy.interpolate import make_lsq_spline, BSpline
>>> t = [-1, 0, 1]
>>> k = 3
>>> t = np.r_[(x[0],)*(k+1),
...           t,
...           (x[-1],)*(k+1)]
>>> spl = make_lsq_spline(x, y, t, k)
```

Example 3 (sql):
```sql
>>> from scipy.interpolate import make_interp_spline
>>> spl_i = make_interp_spline(x, y)
```

Example 4 (unknown):
```unknown
>>> xs = np.linspace(-3, 3, 100)
>>> plt.plot(x, y, 'ro', ms=5)
>>> plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')
>>> plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')
>>> plt.legend(loc='best')
>>> plt.show()
```

---

## cho_solve#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_solve.html

**Contents:**
- cho_solve#

Solve the linear equations A x = b, given the Cholesky factorization of A.

Cholesky factorization of a, as given by cho_factor

Whether to overwrite data in b (may improve performance)

Whether to check that the input matrices contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

The solution to the system A x = b

Cholesky factorization of a matrix

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import cho_factor, cho_solve
>>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
>>> c, low = cho_factor(A)
>>> x = cho_solve((c, low), [1, 1, 1, 1])
>>> np.allclose(A @ x - [1, 1, 1, 1], np.zeros(4))
True
```

---

## griddata#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

**Contents:**
- griddata#

Convenience function for interpolating unstructured data in multiple dimensions.

Data point coordinates.

Points at which to interpolate data.

Method of interpolation. One of

return the value at the data point closest to the point of interpolation. See NearestNDInterpolator for more details.

tessellate the input point set to N-D simplices, and interpolate linearly on each simplex. See LinearNDInterpolator for more details.

return the value determined from a cubic spline.

return the value determined from a piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface. See CloughTocher2DInterpolator for more details.

Value used to fill in for requested points outside of the convex hull of the input points. If not provided, then the default is nan. This option has no effect for the ‘nearest’ method.

Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.

Added in version 0.14.0.

Array of interpolated values.

Piecewise linear interpolator in N dimensions.

Nearest-neighbor interpolator in N dimensions.

Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.

Interpolation on a regular grid or rectilinear grid.

Interpolator on a regular or rectilinear grid in arbitrary dimensions (interpn wraps this class).

Added in version 0.9.

For data on a regular grid use interpn instead.

Suppose we want to interpolate the 2-D function

on a grid in [0, 1]x[0, 1]

but we only know its values at 1000 data points:

This can be done with griddata – below we try out all of the interpolation methods:

One can see that the exact result is reproduced by all of the methods to some degree, but for this smooth function the piecewise cubic interpolant gives the best results:

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> def func(x, y):
...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
```

Example 2 (json):
```json
>>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
```

Example 3 (unknown):
```unknown
>>> rng = np.random.default_rng()
>>> points = rng.random((1000, 2))
>>> values = func(points[:,0], points[:,1])
```

Example 4 (sql):
```sql
>>> from scipy.interpolate import griddata
>>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
>>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
>>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
```

---

## Cython optimize root finding API#

**URL:** https://docs.scipy.org/doc/scipy/reference/optimize.cython_optimize.html

**Contents:**
- Cython optimize root finding API#
- Callback signature#
- Examples#
- Full output#

The underlying C functions for the following root finders can be accessed directly using Cython:

The Cython API for the root finding functions is similar except there is no disp argument. Import the root finding functions using cimport from scipy.optimize.cython_optimize.

The zeros functions in cython_optimize expect a callback that takes a double for the scalar independent variable as the 1st argument and a user defined struct with any extra parameters as the 2nd argument.

Usage of cython_optimize requires Cython to write callbacks that are compiled into C. For more information on compiling Cython, see the Cython Documentation.

These are the basic steps:

Create a Cython .pyx file, for example: myexample.pyx.

Import the desired root finder from cython_optimize.

Write the callback function, and call the selected root finding function passing the callback, any extra arguments, and the other solver parameters.

If you want to call your function from Python, create a Cython wrapper, and a Python function that calls the wrapper, or use cpdef. Then, in Python, you can import and run the example.

Create a Cython .pxd file if you need to export any Cython functions.

The functions in cython_optimize can also copy the full output from the solver to a C struct that is passed as its last argument. If you don’t want the full output, just pass NULL. The full output struct must be type zeros_full_output, which is defined in scipy.optimize.cython_optimize with the following fields:

int funcalls: number of function calls

int iterations: number of iterations

int error_num: error number

double root: root of function

The root is copied by cython_optimize to the full output struct. An error number of -1 means a sign error, -2 means a convergence error, and 0 means the solver converged. Continuing from the previous example:

**Examples:**

Example 1 (sql):
```sql
from scipy.optimize.cython_optimize cimport bisect, ridder, brentq, brenth
```

Example 2 (unknown):
```unknown
double (*callback_type)(double, void*) noexcept
```

Example 3 (python):
```python
from scipy.optimize.cython_optimize cimport brentq

# import math from Cython
from libc cimport math

myargs = {'C0': 1.0, 'C1': 0.7}  # a dictionary of extra arguments
XLO, XHI = 0.5, 1.0  # lower and upper search boundaries
XTOL, RTOL, MITR = 1e-3, 1e-3, 10  # other solver parameters

# user-defined struct for extra parameters
ctypedef struct test_params:
    double C0
    double C1


# user-defined callback
cdef double f(double x, void *args) noexcept:
    cdef test_params *myargs = <test_params *> args
    return myargs.C0 - math.exp(-(x - myargs.C1))


# Cython wrapper function
cdef double brentq_wrapper_example(dict args, double xa, double xb,
                                   double xtol, double rtol, int mitr):
    # Cython automatically casts dictionary to struct
    cdef test_params myargs = args
    return brentq(
        f, xa, xb, <test_params *> &myargs, xtol, rtol, mitr, NULL)


# Python function
def brentq_example(args=myargs, xa=XLO, xb=XHI, xtol=XTOL, rtol=RTOL,
                   mitr=MITR):
    '''Calls Cython wrapper from Python.'''
    return brentq_wrapper_example(args, xa, xb, xtol, rtol, mitr)
```

Example 4 (python):
```python
from myexample import brentq_example

x = brentq_example()
# 0.6999942848231314
```

---

## minimum_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter.html

**Contents:**
- minimum_filter#

Calculate a multidimensional minimum filter.

See footprint, below. Ignored if footprint is given.

Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function. footprint is a boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)). We adjust size to the number of dimensions of the input array, so that, if the input array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When footprint is given, size is ignored.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for size, origin, and/or mode must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

Filtered array. Has the same shape as input.

A sequence of modes (one per axis) is only supported when the footprint is separable. Otherwise, a single mode string must be provided.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = ndimage.minimum_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## generic_filter1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter1d.html

**Contents:**
- generic_filter1d#

Calculate a 1-D filter along the given axis.

generic_filter1d iterates over the lines of the array, calling the given function at each line. The arguments of the line are the input line, and the output line. The input and output lines are 1-D double arrays. The input line is extended appropriately according to the filter size and origin. The output line must be modified in-place with the result.

Function to apply along given axis.

Length of the filter.

The axis of input along which to calculate. Default is -1.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right.

Sequence of extra positional arguments to pass to passed function.

dict of extra keyword arguments to pass to passed function.

Filtered array. Has the same shape as input.

This function also accepts low-level callback functions with one of the following signatures and wrapped in scipy.LowLevelCallable:

The calling function iterates over the lines of the input and output arrays, calling the callback function at each line. The current line is extended according to the border conditions set by the calling function, and the result is copied into the array that is passed through input_line. The length of the input line (after extension) is passed through input_length. The callback function should apply the filter and store the result in the array passed through output_line. The length of the output line is passed through output_length. user_data is the data pointer provided to scipy.LowLevelCallable as-is.

The callback function must return an integer error status that is zero if something went wrong and one otherwise. If an error occurs, you should normally set the python error status with an informative message before returning, otherwise a default error message is set by the calling function.

In addition, some other low-level function pointer specifications are accepted, but these are for backward compatibility only and should not be used in new code.

**Examples:**

Example 1 (r):
```r
int function(double *input_line, npy_intp input_length,
             double *output_line, npy_intp output_length,
             void *user_data)
int function(double *input_line, intptr_t input_length,
             double *output_line, intptr_t output_length,
             void *user_data)
```

---

## correlate1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.correlate1d.html

**Contents:**
- correlate1d#

Calculate a 1-D correlation along the given axis.

The lines of the array along the given axis are correlated with the given weights.

1-D sequence of numbers.

The axis of input along which to calculate. Default is -1.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right.

Correlation result. Has the same shape as input.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.ndimage import correlate1d
>>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
array([ 8, 26,  8, 12,  7, 28, 36,  9])
```

---

## ifft2#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifft2.html

**Contents:**
- ifft2#

Compute the 2-D inverse discrete Fourier Transform.

This function computes the inverse of the 2-D discrete Fourier Transform over any number of axes in an M-D array by means of the Fast Fourier Transform (FFT). In other words, ifft2(fft2(x)) == x to within numerical accuracy. By default, the inverse transform is computed over the last two axes of the input array.

The input, analogously to ifft, should be ordered in the same way as is returned by fft2, i.e., it should have the term for zero frequency in the low-order corner of the two axes, the positive frequency terms in the first half of these axes, the term for the Nyquist frequency in the middle of the axes and the negative frequency terms in the second half of both axes, in order of decreasingly negative frequency.

Input array, can be complex.

Shape (length of each axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). This corresponds to n for ifft(x, n). Along each axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros. if s is not given, the shape of the input along the axes specified by axes is used. See notes for issue on ifft zero padding.

Axes over which to compute the FFT. If not given, the last two axes are used.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axes indicated by axes, or the last two axes if axes is not given.

If s and axes have different length, or axes not given and len(s) != 2.

If an element of axes is larger than the number of axes of x.

The forward 2-D FFT, of which ifft2 is the inverse.

The inverse of the N-D FFT.

ifft2 is just ifftn with a different default for axes.

See ifftn for details and a plotting example, and fft for definition and conventions used.

Zero-padding, analogously with ifft, is performed by appending zeros to the input along the specified dimension. Although this is the common approach, it might lead to surprising results. If another form of zero padding is desired, it must be performed before ifft2 is called.

**Examples:**

Example 1 (json):
```json
>>> import scipy.fft
>>> import numpy as np
>>> x = 4 * np.eye(4)
>>> scipy.fft.ifft2(x)
array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
       [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
       [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
       [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])
```

---

## dst#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html

**Contents:**
- dst#

Return the Discrete Sine Transform of arbitrary type sequence x.

Type of the DST (see Notes). Default type is 2.

Length of the transform. If n < x.shape[axis], x is truncated. If n > x.shape[axis], x is zero-padded. The default results in n = x.shape[axis].

Axis along which the dst is computed; the default is over the last axis (i.e., axis=-1).

Normalization mode (see Notes). Default is “backward”.

If True, the contents of x can be destroyed; the default is False.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

Whether to use the orthogonalized DST variant (see Notes). Defaults to True when norm="ortho" and False otherwise.

Added in version 1.8.0.

The transformed input array.

For type in {2, 3}, norm="ortho" breaks the direct correspondence with the direct Fourier transform. To recover it you must specify orthogonalize=False.

For norm="ortho" both the dst and idst are scaled by the same overall factor in both directions. By default, the transform is also orthogonalized which for types 2 and 3 means the transform definition is modified to give orthogonality of the DST matrix (see below).

For norm="backward", there is no scaling on the dst and the idst is scaled by 1/N where N is the “logical” size of the DST.

There are, theoretically, 8 types of the DST for different combinations of even/odd boundary conditions and boundary off sets [1], only the first 4 types are implemented in SciPy.

There are several definitions of the DST-I; we use the following for norm="backward". DST-I assumes the input is odd around \(n=-1\) and \(n=N\).

Note that the DST-I is only supported for input size > 1. The (unnormalized) DST-I is its own inverse, up to a factor \(2(N+1)\). The orthonormalized DST-I is exactly its own inverse.

orthogonalize has no effect here, as the DST-I matrix is already orthogonal up to a scale factor of 2N.

There are several definitions of the DST-II; we use the following for norm="backward". DST-II assumes the input is odd around \(n=-1/2\) and \(n=N-1/2\); the output is odd around \(k=-1\) and even around \(k=N-1\)

If orthogonalize=True, y[-1] is divided \(\sqrt{2}\) which, when combined with norm="ortho", makes the corresponding matrix of coefficients orthonormal (O @ O.T = np.eye(N)).

There are several definitions of the DST-III, we use the following (for norm="backward"). DST-III assumes the input is odd around \(n=-1\) and even around \(n=N-1\)

If orthogonalize=True, x[-1] is multiplied by \(\sqrt{2}\) which, when combined with norm="ortho", makes the corresponding matrix of coefficients orthonormal (O @ O.T = np.eye(N)).

The (unnormalized) DST-III is the inverse of the (unnormalized) DST-II, up to a factor \(2N\). The orthonormalized DST-III is exactly the inverse of the orthonormalized DST-II.

There are several definitions of the DST-IV, we use the following (for norm="backward"). DST-IV assumes the input is odd around \(n=-0.5\) and even around \(n=N-0.5\)

orthogonalize has no effect here, as the DST-IV matrix is already orthogonal up to a scale factor of 2N.

The (unnormalized) DST-IV is its own inverse, up to a factor \(2N\). The orthonormalized DST-IV is exactly its own inverse.

Wikipedia, “Discrete sine transform”, https://en.wikipedia.org/wiki/Discrete_sine_transform

Compute the DST of a simple 1D array:

This computes the Discrete Sine Transform (DST) of type-II for the input array. The output contains the transformed values corresponding to the given input sequence

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.fft import dst
>>> x = np.array([1, -1, 1, -1])
>>> dst(x, type=2)
array([0., 0., 0., 8.])
```

---

## mmwrite#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html

**Contents:**
- mmwrite#

Writes the sparse or dense array a to Matrix Market file-like target.

Matrix Market filename (extension .mtx) or open file-like object.

Sparse or dense 2-D array.

Comments to be prepended to the Matrix Market file.

Either ‘real’, ‘complex’, ‘pattern’, or ‘integer’.

Number of digits to display for real or complex values.

Either ‘AUTO’, ‘general’, ‘symmetric’, ‘skew-symmetric’, or ‘hermitian’. If symmetry is None the symmetry type of ‘a’ is determined by its values. If symmetry is ‘AUTO’ the symmetry type of ‘a’ is either determined or set to ‘general’, at mmwrite’s discretion.

Changed in version 1.12.0: C++ implementation.

Write a small NumPy array to a matrix market file. The file will be written in the 'array' format.

Add a comment to the output file, and set the precision to 3.

Convert to a sparse matrix before calling mmwrite. This will result in the output format being 'coordinate' rather than 'array'.

Write a complex Hermitian array to a matrix market file. Note that only six values are actually written to the file; the other values are implied by the symmetry.

This method is threaded. The default number of threads is equal to the number of CPUs in the system. Use threadpoolctl to override:

**Examples:**

Example 1 (python):
```python
>>> from io import BytesIO
>>> import numpy as np
>>> from scipy.sparse import coo_array
>>> from scipy.io import mmwrite
```

Example 2 (unknown):
```unknown
>>> a = np.array([[1.0, 0, 0, 0], [0, 2.5, 0, 6.25]])
>>> target = BytesIO()
>>> mmwrite(target, a)
>>> print(target.getvalue().decode('latin1'))
%%MatrixMarket matrix array real general
%
2 4
1
0
0
2.5
0
0
0
6.25
```

Example 3 (unknown):
```unknown
>>> target = BytesIO()
>>> mmwrite(target, a, comment='\n Some test data.\n', precision=3)
>>> print(target.getvalue().decode('latin1'))
%%MatrixMarket matrix array real general
%
% Some test data.
%
2 4
1.00e+00
0.00e+00
0.00e+00
2.50e+00
0.00e+00
0.00e+00
0.00e+00
6.25e+00
```

Example 4 (unknown):
```unknown
>>> target = BytesIO()
>>> mmwrite(target, coo_array(a), precision=3)
>>> print(target.getvalue().decode('latin1'))
%%MatrixMarket matrix coordinate real general
%
2 4 3
1 1 1.00e+00
2 2 2.50e+00
2 4 6.25e+00
```

---

## fftn#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftn.html

**Contents:**
- fftn#

Compute the N-D discrete Fourier Transform.

This function computes the N-D discrete Fourier Transform over any number of axes in an M-D array by means of the Fast Fourier Transform (FFT).

Input array, can be complex.

Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). This corresponds to n for fft(x, n). Along any axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros. if s is not given, the shape of the input along the axes specified by axes is used.

Axes over which to compute the FFT. If not given, the last len(s) axes are used, or all axes if s is also not specified.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axes indicated by axes, or by a combination of s and x, as explained in the parameters section above.

If s and axes have different length.

If an element of axes is larger than the number of axes of x.

The inverse of fftn, the inverse N-D FFT.

The 1-D FFT, with definitions and conventions used.

The N-D FFT of real input.

Shifts zero-frequency terms to centre of array.

The output, analogously to fft, contains the term for zero frequency in the low-order corner of all axes, the positive frequency terms in the first half of all axes, the term for the Nyquist frequency in the middle of all axes and the negative frequency terms in the second half of all axes, in order of decreasingly negative frequency.

**Examples:**

Example 1 (json):
```json
>>> import scipy.fft
>>> import numpy as np
>>> x = np.mgrid[:3, :3, :3][0]
>>> scipy.fft.fftn(x, axes=(1, 2))
array([[[ 0.+0.j,   0.+0.j,   0.+0.j], # may vary
        [ 0.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j]],
       [[ 9.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j]],
       [[18.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j],
        [ 0.+0.j,   0.+0.j,   0.+0.j]]])
>>> scipy.fft.fftn(x, (2, 2), axes=(0, 1))
array([[[ 2.+0.j,  2.+0.j,  2.+0.j], # may vary
        [ 0.+0.j,  0.+0.j,  0.+0.j]],
       [[-2.+0.j, -2.+0.j, -2.+0.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j]]])
```

Example 2 (typescript):
```typescript
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
...                      2 * np.pi * np.arange(200) / 34)
>>> S = np.sin(X) + np.cos(Y) + rng.uniform(0, 1, X.shape)
>>> FS = scipy.fft.fftn(S)
>>> plt.imshow(np.log(np.abs(scipy.fft.fftshift(FS))**2))
<matplotlib.image.AxesImage object at 0x...>
>>> plt.show()
```

---

## invhilbert#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.invhilbert.html

**Contents:**
- invhilbert#

Compute the inverse of the Hilbert matrix of order n.

The entries in the inverse of a Hilbert matrix are integers. When n is greater than 14, some entries in the inverse exceed the upper limit of 64 bit integers. The exact argument provides two options for dealing with these large integers.

The order of the Hilbert matrix.

If False, the data type of the array that is returned is np.float64, and the array is an approximation of the inverse. If True, the array is the exact integer inverse array. To represent the exact inverse when n > 14, the returned array is an object array of long integers. For n <= 14, the exact inverse is returned as an array with data type np.int64.

The data type of the array is np.float64 if exact is False. If exact is True, the data type is either np.int64 (for n <= 14) or object (for n > 14). In the latter case, the objects in the array will be long integers.

Create a Hilbert matrix.

Added in version 0.10.0.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import invhilbert
>>> invhilbert(4)
array([[   16.,  -120.,   240.,  -140.],
       [ -120.,  1200., -2700.,  1680.],
       [  240., -2700.,  6480., -4200.],
       [ -140.,  1680., -4200.,  2800.]])
>>> invhilbert(4, exact=True)
array([[   16,  -120,   240,  -140],
       [ -120,  1200, -2700,  1680],
       [  240, -2700,  6480, -4200],
       [ -140,  1680, -4200,  2800]], dtype=int64)
>>> invhilbert(16)[7,7]
4.2475099528537506e+19
>>> invhilbert(16, exact=True)[7,7]
42475099528537378560
```

---

## fftfreq#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftfreq.html

**Contents:**
- fftfreq#

Return the Discrete Fourier Transform sample frequencies.

The returned float array f contains the frequency bin centers in cycles per unit of the sample spacing (with zero at the start). For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.

Given a window length n and a sample spacing d:

Sample spacing (inverse of the sampling rate). Defaults to 1.

The namespace for the return array. Default is None, where NumPy is used.

The device for the return array. Only valid when xp.fft.fftfreq implements the device parameter.

Array of length n containing the sample frequencies.

**Examples:**

Example 1 (unknown):
```unknown
f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
```

Example 2 (typescript):
```typescript
>>> import numpy as np
>>> import scipy.fft
>>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
>>> fourier = scipy.fft.fft(signal)
>>> n = signal.size
>>> timestep = 0.1
>>> freq = scipy.fft.fftfreq(n, d=timestep)
>>> freq
array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])
```

---

## Linear Algebra (scipy.linalg)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/linalg.html

**Contents:**
- Linear Algebra (scipy.linalg)#
- scipy.linalg vs numpy.linalg#
- numpy.matrix vs 2-D numpy.ndarray#
- Basic routines#
  - Finding the inverse#
  - Solving a linear system#
  - Finding the determinant#
  - Computing norms#
  - Solving linear least-squares problems and pseudo-inverses#
  - Generalized inverse#

When SciPy is built using the optimized ATLAS LAPACK and BLAS libraries, it has very fast linear algebra capabilities. If you dig deep enough, all of the raw LAPACK and BLAS libraries are available for your use for even more speed. In this section, some easier-to-use interfaces to these routines are described.

All of these linear algebra routines expect an object that can be converted into a 2-D array. The output of these routines is also a 2-D array.

scipy.linalg contains all the functions in numpy.linalg. plus some other more advanced ones not contained in numpy.linalg.

Another advantage of using scipy.linalg over numpy.linalg is that it is always compiled with BLAS/LAPACK support, while for NumPy this is optional. Therefore, the SciPy version might be faster depending on how NumPy was installed.

Therefore, unless you don’t want to add scipy as a dependency to your numpy program, use scipy.linalg instead of numpy.linalg.

The classes that represent matrices, and basic operations, such as matrix multiplications and transpose are a part of numpy. For convenience, we summarize the differences between numpy.matrix and numpy.ndarray here.

numpy.matrix is matrix class that has a more convenient interface than numpy.ndarray for matrix operations. This class supports, for example, MATLAB-like creation syntax via the semicolon, has matrix multiplication as default for the * operator, and contains I and T members that serve as shortcuts for inverse and transpose:

Despite its convenience, the use of the numpy.matrix class is discouraged, since it adds nothing that cannot be accomplished with 2-D numpy.ndarray objects, and may lead to a confusion of which class is being used. For example, the above code can be rewritten as:

scipy.linalg operations can be applied equally to numpy.matrix or to 2D numpy.ndarray objects.

The inverse of a matrix \(\mathbf{A}\) is the matrix \(\mathbf{B}\), such that \(\mathbf{AB}=\mathbf{I}\), where \(\mathbf{I}\) is the identity matrix consisting of ones down the main diagonal. Usually, \(\mathbf{B}\) is denoted \(\mathbf{B}=\mathbf{A}^{-1}\) . In SciPy, the matrix inverse of the NumPy array, A, is obtained using linalg.inv (A), or using A.I if A is a Matrix. For example, let

The following example demonstrates this computation in SciPy

Solving linear systems of equations is straightforward using the scipy command linalg.solve. This command expects an input matrix and a right-hand side vector. The solution vector is then computed. An option for entering a symmetric matrix is offered, which can speed up the processing when applicable. As an example, suppose it is desired to solve the following simultaneous equations:

We could find the solution vector using a matrix inverse:

However, it is better to use the linalg.solve command, which can be faster and more numerically stable. In this case, it, however, gives the same answer as shown in the following example:

The determinant of a square matrix \(\mathbf{A}\) is often denoted \(\left|\mathbf{A}\right|\) and is a quantity often used in linear algebra. Suppose \(a_{ij}\) are the elements of the matrix \(\mathbf{A}\) and let \(M_{ij}=\left|\mathbf{A}_{ij}\right|\) be the determinant of the matrix left by removing the \(i^{\textrm{th}}\) row and \(j^{\textrm{th}}\) column from \(\mathbf{A}\) . Then, for any row \(i,\)

This is a recursive way to define the determinant, where the base case is defined by accepting that the determinant of a \(1\times1\) matrix is the only matrix element. In SciPy the determinant can be calculated with linalg.det. For example, the determinant of

In SciPy, this is computed as shown in this example:

Matrix and vector norms can also be computed with SciPy. A wide range of norm definitions are available using different parameters to the order argument of linalg.norm. This function takes a rank-1 (vectors) or a rank-2 (matrices) array and an optional order argument (default is 2). Based on these inputs, a vector or matrix norm of the requested order is computed.

For vector x, the order parameter can be any real number including inf or -inf. The computed norm is

For matrix \(\mathbf{A}\), the only valid values for norm are \(\pm2,\pm1,\) \(\pm\) inf, and ‘fro’ (or ‘f’) Thus,

where \(\sigma_{i}\) are the singular values of \(\mathbf{A}\).

Linear least-squares problems occur in many branches of applied mathematics. In this problem, a set of linear scaling coefficients is sought that allows a model to fit the data. In particular, it is assumed that data \(y_{i}\) is related to data \(\mathbf{x}_{i}\) through a set of coefficients \(c_{j}\) and model functions \(f_{j}\left(\mathbf{x}_{i}\right)\) via the model

where \(\epsilon_{i}\) represents uncertainty in the data. The strategy of least squares is to pick the coefficients \(c_{j}\) to minimize

Theoretically, a global minimum will occur when

When \(\mathbf{A^{H}A}\) is invertible, then

where \(\mathbf{A}^{\dagger}\) is called the pseudo-inverse of \(\mathbf{A}.\) Notice that using this definition of \(\mathbf{A}\) the model can be written

The command linalg.lstsq will solve the linear least-squares problem for \(\mathbf{c}\) given \(\mathbf{A}\) and \(\mathbf{y}\) . In addition, linalg.pinv will find \(\mathbf{A}^{\dagger}\) given \(\mathbf{A}.\)

The following example and figure demonstrate the use of linalg.lstsq and linalg.pinv for solving a data-fitting problem. The data shown below were generated using the model:

where \(x_{i}=0.1i\) for \(i=1\ldots10\) , \(c_{1}=5\), and \(c_{2}=4.\) Noise is added to \(y_{i}\) and the coefficients \(c_{1}\) and \(c_{2}\) are estimated using linear least squares.

The generalized inverse is calculated using the command linalg.pinv. Let \(\mathbf{A}\) be an \(M\times N\) matrix, then if \(M>N\), the generalized inverse is

while if \(M<N\) matrix, the generalized inverse is

In the case that \(M=N\), then

as long as \(\mathbf{A}\) is invertible.

In many applications, it is useful to decompose a matrix using other representations. There are several decompositions supported by SciPy.

The eigenvalue-eigenvector problem is one of the most commonly employed linear algebra operations. In one popular form, the eigenvalue-eigenvector problem is to find for some square matrix \(\mathbf{A}\) scalars \(\lambda\) and corresponding vectors \(\mathbf{v}\), such that

For an \(N\times N\) matrix, there are \(N\) (not necessarily distinct) eigenvalues — roots of the (characteristic) polynomial

The eigenvectors, \(\mathbf{v}\), are also sometimes called right eigenvectors to distinguish them from another set of left eigenvectors that satisfy

With its default optional arguments, the command linalg.eig returns \(\lambda\) and \(\mathbf{v}.\) However, it can also return \(\mathbf{v}_{L}\) and just \(\lambda\) by itself ( linalg.eigvals returns just \(\lambda\) as well).

In addition, linalg.eig can also solve the more general eigenvalue problem

for square matrices \(\mathbf{A}\) and \(\mathbf{B}.\) The standard eigenvalue problem is an example of the general eigenvalue problem for \(\mathbf{B}=\mathbf{I}.\) When a generalized eigenvalue problem can be solved, it provides a decomposition of \(\mathbf{A}\) as

where \(\mathbf{V}\) is the collection of eigenvectors into columns and \(\boldsymbol{\Lambda}\) is a diagonal matrix of eigenvalues.

By definition, eigenvectors are only defined up to a constant scale factor. In SciPy, the scaling factor for the eigenvectors is chosen so that \(\left\Vert \mathbf{v}\right\Vert ^{2}=\sum_{i}v_{i}^{2}=1.\)

As an example, consider finding the eigenvalues and eigenvectors of the matrix

The characteristic polynomial is

The roots of this polynomial are the eigenvalues of \(\mathbf{A}\):

The eigenvectors corresponding to each eigenvalue can be found using the original equation. The eigenvectors associated with these eigenvalues can then be found.

Singular value decomposition (SVD) can be thought of as an extension of the eigenvalue problem to matrices that are not square. Let \(\mathbf{A}\) be an \(M\times N\) matrix with \(M\) and \(N\) arbitrary. The matrices \(\mathbf{A}^{H}\mathbf{A}\) and \(\mathbf{A}\mathbf{A}^{H}\) are square hermitian matrices [1] of size \(N\times N\) and \(M\times M\), respectively. It is known that the eigenvalues of square hermitian matrices are real and non-negative. In addition, there are at most \(\min\left(M,N\right)\) identical non-zero eigenvalues of \(\mathbf{A}^{H}\mathbf{A}\) and \(\mathbf{A}\mathbf{A}^{H}.\) Define these positive eigenvalues as \(\sigma_{i}^{2}.\) The square-root of these are called singular values of \(\mathbf{A}.\) The eigenvectors of \(\mathbf{A}^{H}\mathbf{A}\) are collected by columns into an \(N\times N\) unitary [2] matrix \(\mathbf{V}\), while the eigenvectors of \(\mathbf{A}\mathbf{A}^{H}\) are collected by columns in the unitary matrix \(\mathbf{U}\), the singular values are collected in an \(M\times N\) zero matrix \(\mathbf{\boldsymbol{\Sigma}}\) with main diagonal entries set to the singular values. Then

is the singular value decomposition of \(\mathbf{A}.\) Every matrix has a singular value decomposition. Sometimes, the singular values are called the spectrum of \(\mathbf{A}.\) The command linalg.svd will return \(\mathbf{U}\) , \(\mathbf{V}^{H}\), and \(\sigma_{i}\) as an array of the singular values. To obtain the matrix \(\boldsymbol{\Sigma}\), use linalg.diagsvd. The following example illustrates the use of linalg.svd:

A hermitian matrix \(\mathbf{D}\) satisfies \(\mathbf{D}^{H}=\mathbf{D}.\)

A unitary matrix \(\mathbf{D}\) satisfies \(\mathbf{D}^{H}\mathbf{D}=\mathbf{I}=\mathbf{D}\mathbf{D}^{H}\) so that \(\mathbf{D}^{-1}=\mathbf{D}^{H}.\)

The LU decomposition finds a representation for the \(M\times N\) matrix \(\mathbf{A}\) as

where \(\mathbf{P}\) is an \(M\times M\) permutation matrix (a permutation of the rows of the identity matrix), \(\mathbf{L}\) is in \(M\times K\) lower triangular or trapezoidal matrix ( \(K=\min\left(M,N\right)\)) with unit-diagonal, and \(\mathbf{U}\) is an upper triangular or trapezoidal matrix. The SciPy command for this decomposition is linalg.lu.

Such a decomposition is often useful for solving many simultaneous equations where the left-hand side does not change but the right-hand side does. For example, suppose we are going to solve

for many different \(\mathbf{b}_{i}\). The LU decomposition allows this to be written as

Because \(\mathbf{L}\) is lower-triangular, the equation can be solved for \(\mathbf{U}\mathbf{x}_{i}\) and, finally, \(\mathbf{x}_{i}\) very rapidly using forward- and back-substitution. An initial time spent factoring \(\mathbf{A}\) allows for very rapid solution of similar systems of equations in the future. If the intent for performing LU decomposition is for solving linear systems, then the command linalg.lu_factor should be used followed by repeated applications of the command linalg.lu_solve to solve the system for each new right-hand side.

Cholesky decomposition is a special case of LU decomposition applicable to Hermitian positive definite matrices. When \(\mathbf{A}=\mathbf{A}^{H}\) and \(\mathbf{x}^{H}\mathbf{Ax}\geq0\) for all \(\mathbf{x}\), then decompositions of \(\mathbf{A}\) can be found so that

where \(\mathbf{L}\) is lower triangular and \(\mathbf{U}\) is upper triangular. Notice that \(\mathbf{L}=\mathbf{U}^{H}.\) The command linalg.cholesky computes the Cholesky factorization. For using the Cholesky factorization to solve systems of equations, there are also linalg.cho_factor and linalg.cho_solve routines that work similarly to their LU decomposition counterparts.

The QR decomposition (sometimes called a polar decomposition) works for any \(M\times N\) array and finds an \(M\times M\) unitary matrix \(\mathbf{Q}\) and an \(M\times N\) upper-trapezoidal matrix \(\mathbf{R}\), such that

Notice that if the SVD of \(\mathbf{A}\) is known, then the QR decomposition can be found.

implies that \(\mathbf{Q}=\mathbf{U}\) and \(\mathbf{R}=\boldsymbol{\Sigma}\mathbf{V}^{H}.\) Note, however, that in SciPy independent algorithms are used to find QR and SVD decompositions. The command for QR decomposition is linalg.qr.

For a square \(N\times N\) matrix, \(\mathbf{A}\), the Schur decomposition finds (not necessarily unique) matrices \(\mathbf{T}\) and \(\mathbf{Z}\), such that

where \(\mathbf{Z}\) is a unitary matrix and \(\mathbf{T}\) is either upper triangular or quasi upper triangular, depending on whether or not a real Schur form or complex Schur form is requested. For a real Schur form both \(\mathbf{T}\) and \(\mathbf{Z}\) are real-valued when \(\mathbf{A}\) is real-valued. When \(\mathbf{A}\) is a real-valued matrix, the real Schur form is only quasi upper triangular because \(2\times2\) blocks extrude from the main diagonal corresponding to any complex-valued eigenvalues. The command linalg.schur finds the Schur decomposition, while the command linalg.rsf2csf converts \(\mathbf{T}\) and \(\mathbf{Z}\) from a real Schur form to a complex Schur form. The Schur form is especially useful in calculating functions of matrices.

The following example illustrates the Schur decomposition:

scipy.linalg.interpolative contains routines for computing the interpolative decomposition (ID) of a matrix. For a matrix \(A \in \mathbb{C}^{m \times n}\) of rank \(k \leq \min \{ m, n \}\) this is a factorization

where \(\Pi = [\Pi_{1}, \Pi_{2}]\) is a permutation matrix with \(\Pi_{1} \in \{ 0, 1 \}^{n \times k}\), i.e., \(A \Pi_{2} = A \Pi_{1} T\). This can equivalently be written as \(A = BP\), where \(B = A \Pi_{1}\) and \(P = [I, T] \Pi^{\mathsf{T}}\) are the skeleton and interpolation matrices, respectively.

scipy.linalg.interpolative — for more information.

Consider the function \(f\left(x\right)\) with Taylor series expansion

A matrix function can be defined using this Taylor series for the square matrix \(\mathbf{A}\) as

While this serves as a useful representation of a matrix function, it is rarely the best way to calculate a matrix function. In particular, if the matrix is not diagonalizable, results may be inaccurate.

The matrix exponential is one of the more common matrix functions. The preferred method for implementing the matrix exponential is to use scaling and a Padé approximation for \(e^{x}\). This algorithm is implemented as linalg.expm.

The inverse of the matrix exponential is the matrix logarithm defined as the inverse of the matrix exponential:

The matrix logarithm can be obtained with linalg.logm.

The trigonometric functions, \(\sin\), \(\cos\), and \(\tan\), are implemented for matrices in linalg.sinm, linalg.cosm, and linalg.tanm, respectively. The matrix sine and cosine can be defined using Euler’s identity as

and so the matrix tangent is defined as

The hyperbolic trigonometric functions, \(\sinh\), \(\cosh\), and \(\tanh\), can also be defined for matrices using the familiar definitions:

These matrix functions can be found using linalg.sinhm, linalg.coshm, and linalg.tanhm.

Finally, any arbitrary function that takes one complex number and returns a complex number can be called as a matrix function using the command linalg.funm. This command takes the matrix and an arbitrary Python function. It then implements an algorithm from Golub and Van Loan’s book “Matrix Computations” to compute the function applied to the matrix using a Schur decomposition. Note that the function needs to accept complex numbers as input in order to work with this algorithm. For example, the following code computes the zeroth-order Bessel function applied to a matrix.

Note how, by virtue of how matrix analytic functions are defined, the Bessel function has acted on the matrix eigenvalues.

SciPy and NumPy provide several functions for creating special matrices that are frequently used in engineering and science.

scipy.linalg.block_diag

Create a block diagonal matrix from the provided arrays.

scipy.linalg.circulant

Create a circulant matrix.

scipy.linalg.companion

Create a companion matrix.

scipy.linalg.convolution_matrix

Create a convolution matrix.

Create a discrete Fourier transform matrix.

Create a symmetric Fiedler matrix.

scipy.linalg.fiedler_companion

Create a Fiedler companion matrix.

scipy.linalg.hadamard

Create an Hadamard matrix.

Create a Hankel matrix.

Create a Helmert matrix.

Create a Hilbert matrix.

scipy.linalg.invhilbert

Create the inverse of a Hilbert matrix.

Create a Leslie matrix.

Create a Pascal matrix.

scipy.linalg.invpascal

Create the inverse of a Pascal matrix.

scipy.linalg.toeplitz

Create a Toeplitz matrix.

Create a Van der Monde matrix.

For examples of the use of these functions, see their respective docstrings.

Some of SciPy’s linear algebra functions can process batches of scalars, 1D-, or 2D-arrays given N-d array input. For example, a linear algebra function that typically accepts a (2D) matrix may accept an array of shape (4, 3, 2), which it would interpret as a batch of four 3-by-2 matrices. In this case, we say that the the core shape of the input is (3, 2) and the batch shape is (4,). Likewise, a linear algebra function that typically accepts a (1D) vector would treat a (4, 3, 2) array as a (4, 3) batch of vectors, in which case the core shape of the input is (2,) and the batch shape is (4, 3). The length of the core shape is also referred to as the core dimension. In these cases, the final shape of the output is the batch shape of the input concatenated with the core shape of the output (i.e., the shape of the output when the batch shape of the input is ()). For more information, see Batched Linear Operations.

**Examples:**

Example 1 (json):
```json
>>> import numpy as np
>>> A = np.asmatrix('[1 2;3 4]')
>>> A
matrix([[1, 2],
        [3, 4]])
>>> A.I
matrix([[-2. ,  1. ],
        [ 1.5, -0.5]])
>>> b = np.asmatrix('[5 6]')
>>> b
matrix([[5, 6]])
>>> b.T
matrix([[5],
        [6]])
>>> A*b.T
matrix([[17],
        [39]])
```

Example 2 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> A = np.array([[1,2],[3,4]])
>>> A
array([[1, 2],
      [3, 4]])
>>> linalg.inv(A)
array([[-2. ,  1. ],
      [ 1.5, -0.5]])
>>> b = np.array([[5,6]]) #2D array
>>> b
array([[5, 6]])
>>> b.T
array([[5],
      [6]])
>>> A*b #not matrix multiplication!
array([[ 5, 12],
      [15, 24]])
>>> A.dot(b.T) #matrix multiplication
array([[17],
      [39]])
>>> b = np.array([5,6]) #1D array
>>> b
array([5, 6])
>>> b.T  #not matrix transpose!
array([5, 6])
>>> A.dot(b)  #does not matter for multiplication
array([17, 39])
```

Example 3 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> A = np.array([[1,3,5],[2,5,1],[2,3,8]])
>>> A
array([[1, 3, 5],
      [2, 5, 1],
      [2, 3, 8]])
>>> linalg.inv(A)
array([[-1.48,  0.36,  0.88],
      [ 0.56,  0.08, -0.36],
      [ 0.16, -0.12,  0.04]])
>>> A.dot(linalg.inv(A)) #double check
array([[  1.00000000e+00,  -1.11022302e-16,  -5.55111512e-17],
      [  3.05311332e-16,   1.00000000e+00,   1.87350135e-16],
      [  2.22044605e-16,  -1.11022302e-16,   1.00000000e+00]])
```

Example 4 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> A = np.array([[1, 2], [3, 4]])
>>> A
array([[1, 2],
      [3, 4]])
>>> b = np.array([[5], [6]])
>>> b
array([[5],
      [6]])
>>> linalg.inv(A).dot(b)  # slow
array([[-4. ],
      [ 4.5]])
>>> A.dot(linalg.inv(A).dot(b)) - b  # check
array([[  8.88178420e-16],
      [  2.66453526e-15]])
>>> np.linalg.solve(A, b)  # fast
array([[-4. ],
      [ 4.5]])
>>> A.dot(np.linalg.solve(A, b)) - b  # check
array([[ 0.],
      [ 0.]])
```

---

## fourier_gaussian#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_gaussian.html

**Contents:**
- fourier_gaussian#

Multidimensional Gaussian fourier filter.

The array is multiplied with the fourier transform of a Gaussian kernel.

The sigma of the Gaussian kernel. If a float, sigma is the same for all axes. If a sequence, sigma has to contain one value for each axis.

If n is negative (default), then the input is assumed to be the result of a complex fft. If n is larger than or equal to zero, the input is assumed to be the result of a real fft, and n gives the length of the array before transformation along the real transform direction.

The axis of the real transform.

If given, the result of filtering the input is placed in this array.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import numpy.fft
>>> import matplotlib.pyplot as plt
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plt.gray()  # show the filtered result in grayscale
>>> ascent = datasets.ascent()
>>> input_ = numpy.fft.fft2(ascent)
>>> result = ndimage.fourier_gaussian(input_, sigma=4)
>>> result = numpy.fft.ifft2(result)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result.real)  # the imaginary part is an artifact
>>> plt.show()
```

---

## make_splprep#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splprep.html

**Contents:**
- make_splprep#

Create a smoothing parametric B-spline curve with bounded error, minimizing derivative jumps.

Given a list of N 1D arrays, x, which represent a curve in N-dimensional space parametrized by u, find a smooth approximating spline curve g(u).

Sampled data points representing the curve in ndim dimensions. The typical use is a list of 1D arrays, each of length m.

Strictly positive 1D array of weights. The weights are used in computing the weighted least-squares spline fit. If the errors in the x values have standard deviation given by the vector d, then w should be 1/d. Default is np.ones(m).

An array of parameter values for the curve in the parametric form. If not given, these values are calculated automatically, according to:

The end-points of the parameters interval. Default to u[0] and u[-1].

Degree of the spline. Cubic splines, k=3, are recommended. Even values of k should be avoided especially with a small s value. Default is k=3

A smoothing condition. The amount of smoothness is determined by satisfying the conditions:

where g(u) is the smoothed approximation to x. The user can use s to control the trade-off between closeness and smoothness of fit. Larger s means more smoothing while smaller values of s indicate less smoothing. Recommended values of s depend on the weights, w. If the weights represent the inverse of the standard deviation of x, then a good s value should be found in the range (m - sqrt(2*m), m + sqrt(2*m)), where m is the number of data points in x and w.

The spline knots. If None (default), the knots will be constructed automatically. There must be at least 2*k + 2 and at most m + k + 1 knots.

The target length of the knot vector. Should be between 2*(k + 1) (the minimum number of knots for a degree-k spline), and m + k + 1 (the number of knots of the interpolating spline). The actual number of knots returned by this routine may be slightly larger than nest. Default is None (no limit, add up to m + k + 1 knots).

For s=0, spl(u) == x. For non-zero values of s, spl represents the smoothed approximation to x, generally with fewer knots.

The values of the parameters

is used under the hood for generating the knots

the analog of this routine 1D functions

construct an interpolating spline (s = 0)

construct the least-squares spline given the knot vector

a FITPACK analog of this routine

Given a set of \(m\) data points in \(D\) dimensions, \(\vec{x}_j\), with \(j=1, ..., m\) and \(\vec{x}_j = (x_{j; 1}, ..., x_{j; D})\), this routine constructs the parametric spline curve \(g_a(u)\) with \(a=1, ..., D\), to minimize the sum of jumps, \(D_{i; a}\), of the k-th derivative at the internal knots (\(u_b < t_i < u_e\)), where

Specifically, the routine constructs the spline function \(g(u)\) which minimizes

where \(u_j\) is the value of the parameter corresponding to the data point \((x_{j; 1}, ..., x_{j; D})\), and \(s > 0\) is the input parameter.

In other words, we balance maximizing the smoothness (measured as the jumps of the derivative, the first criterion), and the deviation of \(g(u_j)\) from the data \(x_j\) (the second criterion).

Note that the summation in the second criterion is over all data points, and in the first criterion it is over the internal spline knots (i.e. those with ub < t[i] < ue). The spline knots are in general a subset of data, see generate_knots for details.

Added in version 1.15.0.

P. Dierckx, “Algorithms for smoothing data with periodic and parametric splines, Computer Graphics and Image Processing”, 20 (1982) 171-184.

P. Dierckx, “Curve and surface fitting with splines”, Monographs on Numerical Analysis, Oxford University Press, 1993.

**Examples:**

Example 1 (unknown):
```unknown
v[0] = 0
v[i] = v[i-1] + distance(x[i], x[i-1])
u[i] = v[i] / v[-1]
```

Example 2 (unknown):
```unknown
sum((w * (g(u) - x))**2) <= s,
```

---

## Sparse eigenvalue problems with ARPACK#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/arpack.html

**Contents:**
- Sparse eigenvalue problems with ARPACK#
- Introduction#
- Basic functionality#
- Shift-invert mode#
- Examples#
- Use of LinearOperator#
- References#

ARPACK [1] is a Fortran package which provides routines for quickly finding a few eigenvalues/eigenvectors of large sparse matrices. In order to find these solutions, it requires only left-multiplication by the matrix in question. This operation is performed through a reverse-communication interface. The result of this structure is that ARPACK is able to find eigenvalues and eigenvectors of any linear function mapping a vector to a vector.

All of the functionality provided in ARPACK is contained within the two high-level interfaces scipy.sparse.linalg.eigs and scipy.sparse.linalg.eigsh. eigs provides interfaces for finding the eigenvalues/vectors of real or complex nonsymmetric square matrices, while eigsh provides interfaces for real-symmetric or complex-hermitian matrices.

ARPACK can solve either standard eigenvalue problems of the form

or general eigenvalue problems of the form

The power of ARPACK is that it can compute only a specified subset of eigenvalue/eigenvector pairs. This is accomplished through the keyword which. The following values of which are available:

which = 'LM' : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in the euclidean norm of complex numbers.

which = 'SM' : Eigenvalues with smallest magnitude (eigs, eigsh), that is, smallest eigenvalues in the euclidean norm of complex numbers.

which = 'LR' : Eigenvalues with largest real part (eigs).

which = 'SR' : Eigenvalues with smallest real part (eigs).

which = 'LI' : Eigenvalues with largest imaginary part (eigs).

which = 'SI' : Eigenvalues with smallest imaginary part (eigs).

which = 'LA' : Eigenvalues with largest algebraic value (eigsh), that is, largest eigenvalues inclusive of any negative sign.

which = 'SA' : Eigenvalues with smallest algebraic value (eigsh), that is, smallest eigenvalues inclusive of any negative sign.

which = 'BE' : Eigenvalues from both ends of the spectrum (eigsh).

Note that ARPACK is generally better at finding extremal eigenvalues, that is, eigenvalues with large magnitudes. In particular, using which = 'SM' may lead to slow execution time and/or anomalous results. A better approach is to use shift-invert mode.

Shift-invert mode relies on the following observation. For the generalized eigenvalue problem

Imagine you’d like to find the smallest and largest eigenvalues and the corresponding eigenvectors for a large matrix. ARPACK can handle many forms of input: dense matrices ,such as numpy.ndarray instances, sparse matrices, such as scipy.sparse.csr_matrix, or a general linear operator derived from scipy.sparse.linalg.LinearOperator. For this example, for simplicity, we’ll construct a symmetric, positive-definite matrix.

We now have a symmetric matrix X, with which to test the routines. First, compute a standard eigenvalue decomposition using eigh:

As the dimension of X grows, this routine becomes very slow. Especially, if only a few eigenvectors and eigenvalues are needed, ARPACK can be a better option. First let’s compute the largest eigenvalues (which = 'LM') of X and compare them to the known results:

The results are as expected. ARPACK recovers the desired eigenvalues and they match the previously known results. Furthermore, the eigenvectors are orthogonal, as we’d expect. Now, let’s attempt to solve for the eigenvalues with smallest magnitude:

Oops. We see that, as mentioned above, ARPACK is not quite as adept at finding small eigenvalues. There are a few ways this problem can be addressed. We could increase the tolerance (tol) to lead to faster convergence:

This works, but we lose the precision in the results. Another option is to increase the maximum number of iterations (maxiter) from 1000 to 5000:

We get the results we’d hoped for, but the computation time is much longer. Fortunately, ARPACK contains a mode that allows a quick determination of non-external eigenvalues: shift-invert mode. As mentioned above, this mode involves transforming the eigenvalue problem to an equivalent problem with different eigenvalues. In this case, we hope to find eigenvalues near zero, so we’ll choose sigma = 0. The transformed eigenvalues will then satisfy \(\nu = 1/(\lambda - \sigma) = 1/\lambda\), so our small eigenvalues \(\lambda\) become large eigenvalues \(\nu\).

We get the results we were hoping for, with much less computational time. Note that the transformation from \(\nu \to \lambda\) takes place entirely in the background. The user need not worry about the details.

The shift-invert mode provides more than just a fast way to obtain a few small eigenvalues. Say, you desire to find internal eigenvalues and eigenvectors, e.g., those nearest to \(\lambda = 1\). Simply set sigma = 1 and ARPACK will take care of the rest:

The eigenvalues come out in a different order, but they’re all there. Note that the shift-invert mode requires the internal solution of a matrix inverse. This is taken care of automatically by eigsh and eigs, but the operation can also be specified by the user. See the docstring of scipy.sparse.linalg.eigsh and scipy.sparse.linalg.eigs for details.

We consider now the case where you’d like to avoid creating a dense matrix and use scipy.sparse.linalg.LinearOperator instead. Our first linear operator applies element-wise multiplication between the input vector and a vector \(\mathbf{d}\) provided by the user to the operator itself. This operator mimics a diagonal matrix with the elements of \(\mathbf{d}\) along the main diagonal and it has the main benefit that the forward and adjoint operations are simple element-wise multiplications other than matrix-vector multiplications. For a diagonal matrix, we expect the eigenvalues to be equal to the elements along the main diagonal, in this case \(\mathbf{d}\). The eigenvalues and eigenvectors obtained with eigsh are compared to those obtained by using eigh when applied to the dense matrix:

In this case, we have created a quick and easy Diagonal operator. The external library PyLops provides similar capabilities in the Diagonal operator, as well as several other operators.

Finally, we consider a linear operator that mimics the application of a first-derivative stencil. In this case, the operator is equivalent to a real nonsymmetric matrix. Once again, we compare the estimated eigenvalues and eigenvectors with those from a dense matrix that applies the same first derivative to an input signal:

Note that the eigenvalues of this operator are all imaginary. Moreover, the keyword which='LI' of scipy.sparse.linalg.eigs produces the eigenvalues with largest absolute imaginary part (both positive and negative). Again, a more advanced implementation of the first-derivative operator is available in the PyLops library under the name of FirstDerivative operator.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import eig, eigh
>>> from scipy.sparse.linalg import eigs, eigsh
>>> np.set_printoptions(suppress=True)
>>> rng = np.random.default_rng()
>>>
>>> X = rng.random((100, 100)) - 0.5
>>> X = np.dot(X, X.T)  # create a symmetric matrix
```

Example 2 (unknown):
```unknown
>>> evals_all, evecs_all = eigh(X)
```

Example 3 (json):
```json
>>> evals_large, evecs_large = eigsh(X, 3, which='LM')
>>> print(evals_all[-3:])
[29.22435321 30.05590784 30.58591252]
>>> print(evals_large)
[29.22435321 30.05590784 30.58591252]
>>> print(np.dot(evecs_large.T, evecs_all[:,-3:]))
array([[-1.  0.  0.],       # may vary (signs)
       [ 0.  1.  0.],
       [-0.  0. -1.]])
```

Example 4 (unknown):
```unknown
>>> evals_small, evecs_small = eigsh(X, 3, which='SM')
Traceback (most recent call last):       # may vary (convergence)
...
scipy.sparse.linalg._eigen.arpack.arpack.ArpackNoConvergence:
ARPACK error -1: No convergence (1001 iterations, 0/3 eigenvectors converged)
```

---

## Legacy discrete Fourier transforms (scipy.fftpack)#

**URL:** https://docs.scipy.org/doc/scipy/reference/fftpack.html

**Contents:**
- Legacy discrete Fourier transforms (scipy.fftpack)#
- Fast Fourier Transforms (FFTs)#
- Differential and pseudo-differential operators#
- Helper functions#
- Convolutions (scipy.fftpack.convolve)#

This submodule is considered legacy and will no longer receive updates. While we currently have no plans to remove it, we recommend that new code uses more modern alternatives instead. New code should use scipy.fft.

fft(x[, n, axis, overwrite_x])

Return discrete Fourier transform of real or complex sequence.

ifft(x[, n, axis, overwrite_x])

Return discrete inverse Fourier transform of real or complex sequence.

fft2(x[, shape, axes, overwrite_x])

2-D discrete Fourier transform.

ifft2(x[, shape, axes, overwrite_x])

2-D discrete inverse Fourier transform of real or complex sequence.

fftn(x[, shape, axes, overwrite_x])

Return multidimensional discrete Fourier transform.

ifftn(x[, shape, axes, overwrite_x])

Return inverse multidimensional discrete Fourier transform.

rfft(x[, n, axis, overwrite_x])

Discrete Fourier transform of a real sequence.

irfft(x[, n, axis, overwrite_x])

Return inverse discrete Fourier transform of real sequence x.

dct(x[, type, n, axis, norm, overwrite_x])

Return the Discrete Cosine Transform of arbitrary type sequence x.

idct(x[, type, n, axis, norm, overwrite_x])

Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

dctn(x[, type, shape, axes, norm, overwrite_x])

Return multidimensional Discrete Cosine Transform along the specified axes.

idctn(x[, type, shape, axes, norm, overwrite_x])

Return multidimensional Discrete Cosine Transform along the specified axes.

dst(x[, type, n, axis, norm, overwrite_x])

Return the Discrete Sine Transform of arbitrary type sequence x.

idst(x[, type, n, axis, norm, overwrite_x])

Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

dstn(x[, type, shape, axes, norm, overwrite_x])

Return multidimensional Discrete Sine Transform along the specified axes.

idstn(x[, type, shape, axes, norm, overwrite_x])

Return multidimensional Discrete Sine Transform along the specified axes.

diff(x[, order, period, _cache])

Return kth derivative (or integral) of a periodic sequence x.

tilbert(x, h[, period, _cache])

Return h-Tilbert transform of a periodic sequence x.

itilbert(x, h[, period, _cache])

Return inverse h-Tilbert transform of a periodic sequence x.

Return Hilbert transform of a periodic sequence x.

ihilbert(x[, _cache])

Return inverse Hilbert transform of a periodic sequence x.

cs_diff(x, a, b[, period, _cache])

Return (a,b)-cosh/sinh pseudo-derivative of a periodic sequence.

sc_diff(x, a, b[, period, _cache])

Return (a,b)-sinh/cosh pseudo-derivative of a periodic sequence x.

ss_diff(x, a, b[, period, _cache])

Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.

cc_diff(x, a, b[, period, _cache])

Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.

shift(x, a[, period, _cache])

Shift periodic sequence x by a: y(u) = x(u+a).

Shift the zero-frequency component to the center of the spectrum.

The inverse of fftshift.

fftfreq(n[, d, device])

Return the Discrete Fourier Transform sample frequencies.

DFT sample frequencies (for usage with rfft, irfft).

next_fast_len(target)

Find the next fast size of input data to fft, for zero-padding, etc.

Note that fftshift, ifftshift and fftfreq are numpy functions exposed by fftpack; importing them from numpy should be preferred.

convolve(x,omega,[swap_real_imag,overwrite_x])

Wrapper for convolve.

convolve_z(x,omega_real,omega_imag,[overwrite_x])

Wrapper for convolve_z.

init_convolution_kernel(...)

Wrapper for init_convolution_kernel.

destroy_convolve_cache()

---

## RegularGridInterpolator#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

**Contents:**
- RegularGridInterpolator#

Interpolator of specified order on a rectilinear grid in N ≥ 1 dimensions.

The data must be defined on a rectilinear grid; that is, a rectangular grid with even or uneven spacing. Linear, nearest-neighbor, spline interpolations are supported. After setting up the interpolator object, the interpolation method may be chosen at each evaluation.

The points defining the regular grid in n dimensions. The points in each dimension (i.e. every elements of the points tuple) must be strictly ascending or descending.

The data on the regular grid in n dimensions. Complex data is accepted.

The method of interpolation to perform. Supported are “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”. This parameter will become the default for the object’s __call__ method. Default is “linear”.

If True, when interpolated values are requested outside of the domain of the input data, a ValueError is raised. If False, then fill_value is used. Default is True.

The value to use for points outside of the interpolation domain. If None, values outside the domain are extrapolated. Default is np.nan.

Only used for methods “slinear”, “cubic” and “quintic”. Sparse linear algebra solver for construction of the NdBSpline instance. Default is the iterative solver scipy.sparse.linalg.gcrotmk.

Added in version 1.13.

Additional arguments to pass to solver, if any.

Added in version 1.13.

The points defining the regular grid in n dimensions. This tuple defines the full grid via np.meshgrid(*grid, indexing='ij')

Data values at the grid.

Interpolation method.

Use this value for out-of-bounds arguments to __call__.

If True, out-of-bounds argument raise a ValueError.

__call__(xi[, method, nu])

Interpolation at coordinates.

Nearest neighbor interpolator on unstructured data in N dimensions

Piecewise linear interpolator on unstructured data in N dimensions

a convenience function which wraps RegularGridInterpolator

interpolation on grids with equal spacing (suitable for e.g., N-D image resampling)

Contrary to LinearNDInterpolator and NearestNDInterpolator, this class avoids expensive triangulation of the input data by taking advantage of the regular grid structure.

In other words, this class assumes that the data is defined on a rectilinear grid.

Added in version 0.14.

The ‘slinear’(k=1), ‘cubic’(k=3), and ‘quintic’(k=5) methods are tensor-product spline interpolators, where k is the spline degree, If any dimension has fewer points than k + 1, an error will be raised.

Added in version 1.9.

If the input data is such that dimensions have incommensurate units and differ by many orders of magnitude, the interpolant may have numerical artifacts. Consider rescaling the data before interpolating.

Choosing a solver for spline methods

Spline methods, “slinear”, “cubic” and “quintic” involve solving a large sparse linear system at instantiation time. Depending on data, the default solver may or may not be adequate. When it is not, you may need to experiment with an optional solver argument, where you may choose between the direct solver (scipy.sparse.linalg.spsolve) or iterative solvers from scipy.sparse.linalg. You may need to supply additional parameters via the optional solver_args parameter (for instance, you may supply the starting value or target tolerance). See the scipy.sparse.linalg documentation for the full list of available options.

Alternatively, you may instead use the legacy methods, “slinear_legacy”, “cubic_legacy” and “quintic_legacy”. These methods allow faster construction but evaluations will be much slower.

Rounding rule at half points with `nearest` method

The rounding rule with the nearest method at half points is rounding down.

Python package regulargrid by Johannes Buchner, see https://pypi.python.org/pypi/regulargrid/

Wikipedia, “Trilinear interpolation”, https://en.wikipedia.org/wiki/Trilinear_interpolation

Weiser, Alan, and Sergio E. Zarantonello. “A note on piecewise linear and multilinear table interpolation in many dimensions.” MATH. COMPUT. 50.181 (1988): 189-196. https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf DOI:10.1090/S0025-5718-1988-0917826-0

Evaluate a function on the points of a 3-D grid

As a first example, we evaluate a simple example function on the points of a 3-D grid:

data is now a 3-D array with data[i, j, k] = f(x[i], y[j], z[k]). Next, define an interpolating function from this data:

Evaluate the interpolating function at the two points (x,y,z) = (2.1, 6.2, 8.3) and (3.3, 5.2, 7.1):

which is indeed a close approximation to

Interpolate and extrapolate a 2D dataset

As a second example, we interpolate and extrapolate a 2D data set:

Evaluate and plot the interpolator on a finer grid

Other examples are given in the tutorial.

**Examples:**

Example 1 (python):
```python
>>> from scipy.interpolate import RegularGridInterpolator
>>> import numpy as np
>>> def f(x, y, z):
...     return 2 * x**3 + 3 * y**2 - z
>>> x = np.linspace(1, 4, 11)
>>> y = np.linspace(4, 7, 22)
>>> z = np.linspace(7, 9, 33)
>>> xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
>>> data = f(xg, yg, zg)
```

Example 2 (unknown):
```unknown
>>> interp = RegularGridInterpolator((x, y, z), data)
```

Example 3 (unknown):
```unknown
>>> pts = np.array([[2.1, 6.2, 8.3],
...                 [3.3, 5.2, 7.1]])
>>> interp(pts)
array([ 125.80469388,  146.30069388])
```

Example 4 (unknown):
```unknown
>>> f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)
(125.54200000000002, 145.894)
```

---

## Spatial algorithms and data structures (scipy.spatial)#

**URL:** https://docs.scipy.org/doc/scipy/reference/spatial.html

**Contents:**
- Spatial algorithms and data structures (scipy.spatial)#
- Spatial transformations#
- Nearest-neighbor queries#
- Distance metrics#
- Delaunay triangulation, convex hulls, and Voronoi diagrams#
- Plotting helpers#
- Simplex representation#
  - Functions#
  - Warnings / Errors used in scipy.spatial#

These are contained in the scipy.spatial.transform submodule.

KDTree(data[, leafsize, compact_nodes, ...])

kd-tree for quick nearest-neighbor lookup.

cKDTree(data[, leafsize, compact_nodes, ...])

kd-tree for quick nearest-neighbor lookup

Rectangle(maxes, mins)

Hyperrectangle class.

Distance metrics are contained in the scipy.spatial.distance submodule.

Delaunay(points[, furthest_site, ...])

Delaunay tessellation in N dimensions.

ConvexHull(points[, incremental, qhull_options])

Convex hulls in N dimensions.

Voronoi(points[, furthest_site, ...])

Voronoi diagrams in N dimensions.

SphericalVoronoi(points[, radius, center, ...])

Voronoi diagrams on the surface of a sphere.

HalfspaceIntersection(halfspaces, interior_point)

Halfspace intersections in N dimensions.

delaunay_plot_2d(tri[, ax])

Plot the given Delaunay triangulation in 2-D

convex_hull_plot_2d(hull[, ax])

Plot the given convex hull diagram in 2-D

voronoi_plot_2d(vor[, ax])

Plot the given Voronoi diagram in 2-D

The simplices (triangles, tetrahedra, etc.) appearing in the Delaunay tessellation (N-D simplices), convex hull facets, and Voronoi ridges (N-1-D simplices) are represented in the following scheme:

For Delaunay triangulations and convex hulls, the neighborhood structure of the simplices satisfies the condition: tess.neighbors[i,j] is the neighboring simplex of the ith simplex, opposite to the j-vertex. It is -1 in case of no neighbor.

Convex hull facets also define a hyperplane equation:

Similar hyperplane equations for the Delaunay triangulation correspond to the convex hull facets on the corresponding N+1-D paraboloid.

The Delaunay triangulation objects offer a method for locating the simplex containing a given point, and barycentric coordinate computations.

Find simplices containing the given points.

distance_matrix(x, y[, p, threshold])

Compute the distance matrix.

minkowski_distance(x, y[, p])

Compute the L**p distance between two arrays.

minkowski_distance_p(x, y[, p])

Compute the pth power of the L**p distance between two arrays.

procrustes(data1, data2)

Procrustes analysis, a similarity test for two data sets.

geometric_slerp(start, end, t[, tol])

Geometric spherical linear interpolation.

Raised when Qhull encounters an error condition, such as geometrical degeneracy when options to resolve are not enabled.

**Examples:**

Example 1 (markdown):
```markdown
tess = Delaunay(points)
hull = ConvexHull(points)
voro = Voronoi(points)

# coordinates of the jth vertex of the ith simplex
tess.points[tess.simplices[i, j], :]        # tessellation element
hull.points[hull.simplices[i, j], :]        # convex hull facet
voro.vertices[voro.ridge_vertices[i, j], :] # ridge between Voronoi cells
```

Example 2 (unknown):
```unknown
(hull.equations[i,:-1] * coord).sum() + hull.equations[i,-1] == 0
```

---

## PchipInterpolator#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html

**Contents:**
- PchipInterpolator#

PCHIP shape-preserving interpolator (C1 smooth).

x and y are arrays of values used to approximate some function f, with y = f(x). The interpolant uses monotonic cubic splines to find the value of new points. (PCHIP stands for Piecewise Cubic Hermite Interpolating Polynomial).

A 1-D array of monotonically increasing real values. x cannot include duplicate values (otherwise f is overspecified)

A N-D array of real values. y’s length along the interpolation axis must be equal to the length of x. Use the axis parameter to select the interpolation axis.

Axis in the y array corresponding to the x-coordinate values. Defaults to axis=0.

Whether to extrapolate to out-of-bounds points based on first and last intervals, or to return NaNs.

__call__(x[, nu, extrapolate])

Evaluate the piecewise polynomial or its derivative.

Construct a new piecewise polynomial representing the derivative.

Construct a new piecewise polynomial representing the antiderivative.

integrate(a, b[, extrapolate])

Compute a definite integral over a piecewise polynomial.

solve([y, discontinuity, extrapolate])

Find real solutions of the equation pp(x) == y.

roots([discontinuity, extrapolate])

Find real roots of the piecewise polynomial.

Piecewise-cubic interpolator.

Akima 1D interpolator.

Cubic spline data interpolator.

Piecewise polynomial in terms of coefficients and breakpoints.

The interpolator preserves monotonicity in the interpolation data and does not overshoot if the data is not smooth.

The first derivatives are guaranteed to be continuous, but the second derivatives may jump at \(x_k\).

Determines the derivatives at the points \(x_k\), \(f'_k\), by using PCHIP algorithm [1].

Let \(h_k = x_{k+1} - x_k\), and \(d_k = (y_{k+1} - y_k) / h_k\) are the slopes at internal points \(x_k\). If the signs of \(d_k\) and \(d_{k-1}\) are different or either of them equals zero, then \(f'_k = 0\). Otherwise, it is given by the weighted harmonic mean

where \(w_1 = 2 h_k + h_{k-1}\) and \(w_2 = h_k + 2 h_{k-1}\).

The end slopes are set using a one-sided scheme [2].

F. N. Fritsch and J. Butland, A method for constructing local monotone piecewise cubic interpolants, SIAM J. Sci. Comput., 5(2), 300-304 (1984). DOI:10.1137/0905021.

see, e.g., C. Moler, Numerical Computing with Matlab, 2004. DOI:10.1137/1.9780898717952

---

## leslie#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.leslie.html

**Contents:**
- leslie#

Create a Leslie matrix.

Given the length n array of fecundity coefficients f and the length n-1 array of survival coefficients s, return the associated Leslie matrix.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

The “fecundity” coefficients.

The “survival” coefficients. The length of s must be one less than the length of f, and it must be at least 1.

The array is zero except for the first row, which is f, and the first sub-diagonal, which is s. The data-type of the array will be the data-type of f[0]+s[0].

The Leslie matrix is used to model discrete-time, age-structured population growth [1] [2]. In a population with n age classes, two sets of parameters define a Leslie matrix: the n “fecundity coefficients”, which give the number of offspring per-capita produced by each age class, and the n - 1 “survival coefficients”, which give the per-capita survival rate of each age class.

P. H. Leslie, On the use of matrices in certain population mathematics, Biometrika, Vol. 33, No. 3, 183–212 (Nov. 1945)

P. H. Leslie, Some further notes on the use of matrices in population mathematics, Biometrika, Vol. 35, No. 3/4, 213–245 (Dec. 1948)

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import leslie
>>> leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7])
array([[ 0.1,  2. ,  1. ,  0.1],
       [ 0.2,  0. ,  0. ,  0. ],
       [ 0. ,  0.8,  0. ,  0. ],
       [ 0. ,  0. ,  0.7,  0. ]])
```

---

## netcdf_file#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.html

**Contents:**
- netcdf_file#

A file object for NetCDF data.

A netcdf_file object has two standard attributes: dimensions and variables. The values of both are dictionaries, mapping dimension names to their associated lengths and variable names to variables, respectively. Application programs should never modify these dictionaries.

All other attributes correspond to global attributes defined in the NetCDF file. Global file attributes are created by assigning to an attribute of the netcdf_file object.

read-write-append mode, default is ‘r’

Whether to mmap filename when reading. Default is True when filename is a file name, False when filename is a file-like object. Note that when mmap is in use, data arrays returned refer directly to the mmapped data on disk, and the file cannot be closed as long as references to it exist.

version of netcdf to read / write, where 1 means Classic format and 2 means 64-bit offset format. Default is 1. See here for more info.

Whether to automatically scale and/or mask data based on attributes. Default is False.

Closes the NetCDF file.

createDimension(name, length)

Adds a dimension to the Dimension section of the NetCDF data structure.

createVariable(name, type, dimensions)

Create an empty variable for the netcdf_file object, specifying its data type and the dimensions it uses.

Perform a sync-to-disk flush if the netcdf_file object is in write mode.

Perform a sync-to-disk flush if the netcdf_file object is in write mode.

This module is derived from pupynere. The major advantage of this module over other modules is that it doesn’t require the code to be linked to the NetCDF libraries. However, for a more recent version of the NetCDF standard and additional features, please consider the permissively-licensed netcdf4-python.

NetCDF files are a self-describing binary data format. The file contains metadata that describes the dimensions and variables in the file. More details about NetCDF files can be found here. There are three main sections to a NetCDF data structure:

The dimensions section records the name and length of each dimension used by the variables. The variables would then indicate which dimensions it uses and any attributes such as data units, along with containing the data values for the variable. It is good practice to include a variable that is the same name as a dimension to provide the values for that axes. Lastly, the attributes section would contain additional information such as the name of the file creator or the instrument used to collect the data.

When writing data to a NetCDF file, there is often the need to indicate the ‘record dimension’. A record dimension is the unbounded dimension for a variable. For example, a temperature variable may have dimensions of latitude, longitude and time. If one wants to add more temperature data to the NetCDF file as time progresses, then the temperature variable should have the time dimension flagged as the record dimension.

In addition, the NetCDF file header contains the position of the data in the file, so access can be done in an efficient manner without loading unnecessary data into memory. It uses the mmap module to create Numpy arrays mapped to the data on disk, for the same purpose.

Note that when netcdf_file is used to open a file with mmap=True (default for read-only), arrays returned by it refer to data directly on the disk. The file should not be closed, and cannot be cleanly closed when asked, if such arrays are alive. You may want to copy data arrays obtained from mmapped Netcdf file if they are to be processed after the file is closed, see the example below.

To create a NetCDF file:

Note the assignment of arange(10) to time[:]. Exposing the slice of the time variable allows for the data to be set in the object, rather than letting arange(10) overwrite the time variable.

To read the NetCDF file we just created:

NetCDF files, when opened read-only, return arrays that refer directly to memory-mapped data on disk:

If the data is to be processed after the file is closed, it needs to be copied to main memory:

A NetCDF file can also be used as context manager:

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.io import netcdf_file
>>> import numpy as np
>>> f = netcdf_file('simple.nc', 'w')
>>> f.history = 'Created for a test'
>>> f.createDimension('time', 10)
>>> time = f.createVariable('time', 'i', ('time',))
>>> time[:] = np.arange(10)
>>> time.units = 'days since 2008-01-01'
>>> f.close()
```

Example 2 (python):
```python
>>> from scipy.io import netcdf_file
>>> f = netcdf_file('simple.nc', 'r')
>>> print(f.history)
b'Created for a test'
>>> time = f.variables['time']
>>> print(time.units)
b'days since 2008-01-01'
>>> print(time.shape)
(10,)
>>> print(time[-1])
9
```

Example 3 (unknown):
```unknown
>>> data = time[:]
```

Example 4 (unknown):
```unknown
>>> data = time[:].copy()
>>> del time
>>> f.close()
>>> data.mean()
4.5
```

---

## Parallel execution support in SciPy#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/parallel_execution.html

**Contents:**
- Parallel execution support in SciPy#

SciPy aims to provide functionality that is performant, i.e. has good execution speed. On modern computing hardware, CPUs often have many CPU cores - and hence users may benefit from parallel execution. This page aims to give a brief overview of the options available to employ parallel execution.

Some key points related to parallelism:

SciPy itself defaults to single-threaded execution.

The exception to that single-threaded default is code that calls into a BLAS or LAPACK library for linear algebra functionality (either direct or via NumPy). BLAS/LAPACK libraries almost always default to multi-threaded execution, typically using all available CPU cores.

Users can control the threading behavior of the BLAS/LAPACK library that SciPy and NumPy are linked with through threadpoolctl.

SciPy functionality may provide parallel execution in an opt-in manner. This is exposed through a workers= keyword in individual APIs, which takes an integer for the number of threads or processes to use, and in some cases also a map-like callable (e.g., multiprocessing.Pool). See scipy.fft.fft and scipy.optimize.differential_evolution for examples.

SciPy-internal threading is done with OS-level thread pools. OpenMP is not used within SciPy.

SciPy works well with multiprocessing and with threading. The former has higher overhead than the latter, but is widely used and robust. The latter may offer performance benefits for some usage scenarios - however, please read Thread Safety in SciPy.

SciPy has experimental support for free-threaded CPython, starting with SciPy 1.15.0 (and Python 3.13.0, NumPy 2.1.0).

SciPy has experimental support in a growing number of submodules and functions for array libraries other than NumPy, such as PyTorch, CuPy and JAX. Those libraries default to parallel execution and may offer significant performance benefits (and GPU execution). See Support for the array API standard for more details.

---

## Linear algebra (scipy.linalg)#

**URL:** https://docs.scipy.org/doc/scipy/reference/linalg.html

**Contents:**
- Linear algebra (scipy.linalg)#
- Basics#
- Eigenvalue Problems#
- Decompositions#
- Matrix Functions#
- Matrix Equation Solvers#
- Sketches and Random Projections#
- Special Matrices#
- Low-level routines#

Linear algebra functions.

numpy.linalg for more linear algebra functions. Note that although scipy.linalg imports most of them, identically named functions from scipy.linalg may offer more or slightly differing functionality.

inv(a[, overwrite_a, check_finite])

Compute the inverse of a matrix.

solve(a, b[, lower, overwrite_a, ...])

Solve the equation a @ x = b for x, where a is a square matrix.

solve_banded(l_and_u, ab, b[, overwrite_ab, ...])

Solve the equation a @ x = b for x, where a is the banded matrix defined by ab.

solveh_banded(ab, b[, overwrite_ab, ...])

Solve the equation a @ x = b for x, where a is the Hermitian positive-definite banded matrix defined by ab.

solve_circulant(c, b[, singular, tol, ...])

Solve the equation C @ x = b for x, where C is a circulant matrix defined by c.

solve_triangular(a, b[, trans, lower, ...])

Solve the equation a @ x = b for x, where a is a triangular matrix.

solve_toeplitz(c_or_cr, b[, check_finite])

Solve the equation T @ x = b for x, where T is a Toeplitz matrix defined by c_or_cr.

matmul_toeplitz(c_or_cr, x[, check_finite, ...])

Efficient Toeplitz Matrix-Matrix Multiplication using FFT

det(a[, overwrite_a, check_finite])

Compute the determinant of a matrix

norm(a[, ord, axis, keepdims, check_finite])

Matrix or vector norm.

lstsq(a, b[, cond, overwrite_a, ...])

Compute least-squares solution to the equation a @ x = b.

pinv(a, *[, atol, rtol, return_rank, ...])

Compute the (Moore-Penrose) pseudo-inverse of a matrix.

pinvh(a[, atol, rtol, lower, return_rank, ...])

Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

orthogonal_procrustes(A, B[, check_finite])

Compute the matrix solution of the orthogonal (or unitary) Procrustes problem.

matrix_balance(A[, permute, scale, ...])

Compute a diagonal similarity transformation for row/column balancing.

subspace_angles(A, B)

Compute the subspace angles between two matrices.

Return the lower and upper bandwidth of a 2D numeric array.

issymmetric(a[, atol, rtol])

Check if a square 2D array is symmetric.

ishermitian(a[, atol, rtol])

Check if a square 2D array is Hermitian.

Generic Python-exception-derived object raised by linalg functions.

The warning emitted when a linear algebra related operation is close to fail conditions of the algorithm or loss of accuracy is expected.

eig(a[, b, left, right, overwrite_a, ...])

Solve an ordinary or generalized eigenvalue problem of a square matrix.

eigvals(a[, b, overwrite_a, check_finite, ...])

Compute eigenvalues from an ordinary or generalized eigenvalue problem.

eigh(a[, b, lower, eigvals_only, ...])

Solve a standard or generalized eigenvalue problem for a complex Hermitian or real symmetric matrix.

eigvalsh(a[, b, lower, overwrite_a, ...])

Solves a standard or generalized eigenvalue problem for a complex Hermitian or real symmetric matrix.

eig_banded(a_band[, lower, eigvals_only, ...])

Solve real symmetric or complex Hermitian band matrix eigenvalue problem.

eigvals_banded(a_band[, lower, ...])

Solve real symmetric or complex Hermitian band matrix eigenvalue problem.

eigh_tridiagonal(d, e[, eigvals_only, ...])

Solve eigenvalue problem for a real symmetric tridiagonal matrix.

eigvalsh_tridiagonal(d, e[, select, ...])

Solve eigenvalue problem for a real symmetric tridiagonal matrix.

lu(a[, permute_l, overwrite_a, ...])

Compute LU decomposition of a matrix with partial pivoting.

lu_factor(a[, overwrite_a, check_finite])

Compute pivoted LU decomposition of a matrix.

lu_solve(lu_and_piv, b[, trans, ...])

Solve an equation system, a x = b, given the LU factorization of a

svd(a[, full_matrices, compute_uv, ...])

Singular Value Decomposition.

svdvals(a[, overwrite_a, check_finite])

Compute singular values of a matrix.

Construct the sigma matrix in SVD from singular values and size M, N.

Construct an orthonormal basis for the range of A using SVD

null_space(A[, rcond, overwrite_a, ...])

Construct an orthonormal basis for the null space of A using SVD

ldl(A[, lower, hermitian, overwrite_a, ...])

Computes the LDLt or Bunch-Kaufman factorization of a symmetric/ hermitian matrix.

cholesky(a[, lower, overwrite_a, check_finite])

Compute the Cholesky decomposition of a matrix.

cholesky_banded(ab[, overwrite_ab, lower, ...])

Cholesky decompose a banded Hermitian positive-definite matrix

cho_factor(a[, lower, overwrite_a, check_finite])

Compute the Cholesky decomposition of a matrix, to use in cho_solve

cho_solve(c_and_lower, b[, overwrite_b, ...])

Solve the linear equations A x = b, given the Cholesky factorization of A.

cho_solve_banded(cb_and_lower, b[, ...])

Solve the linear equations A x = b, given the Cholesky factorization of the banded Hermitian A.

Compute the polar decomposition.

qr(a[, overwrite_a, lwork, mode, pivoting, ...])

Compute QR decomposition of a matrix.

qr_multiply(a, c[, mode, pivoting, ...])

Calculate the QR decomposition and multiply Q with a matrix.

qr_update(Q, R, u, v[, overwrite_qruv, ...])

qr_delete(Q, R, k, int p=1[, which, ...])

QR downdate on row or column deletions

qr_insert(Q, R, u, k[, which, rcond, ...])

QR update on row or column insertions

rq(a[, overwrite_a, lwork, mode, check_finite])

Compute RQ decomposition of a matrix.

qz(A, B[, output, lwork, sort, overwrite_a, ...])

QZ decomposition for generalized eigenvalues of a pair of matrices.

ordqz(A, B[, sort, output, overwrite_a, ...])

QZ decomposition for a pair of matrices with reordering.

schur(a[, output, lwork, overwrite_a, sort, ...])

Compute Schur decomposition of a matrix.

rsf2csf(T, Z[, check_finite])

Convert real Schur form to complex Schur form.

hessenberg(a[, calc_q, overwrite_a, ...])

Compute Hessenberg form of a matrix.

Converts complex eigenvalues w and eigenvectors v to real eigenvalues in a block diagonal form wr and the associated real eigenvectors vr, such that.

cossin(X[, p, q, separate, swap_sign, ...])

Compute the cosine-sine (CS) decomposition of an orthogonal/unitary matrix.

scipy.linalg.interpolative – Interpolative matrix decompositions

Compute the matrix exponential of an array.

Compute matrix logarithm.

Compute the matrix cosine.

Compute the matrix sine.

Compute the matrix tangent.

Compute the hyperbolic matrix cosine.

Compute the hyperbolic matrix sine.

Compute the hyperbolic matrix tangent.

Matrix sign function.

sqrtm(A[, disp, blocksize])

Compute, if exists, the matrix square root.

funm(A, func[, disp])

Evaluate a matrix function specified by a callable.

expm_frechet(A, E[, method, compute_expm, ...])

Frechet derivative of the matrix exponential of A in the direction E.

expm_cond(A[, check_finite])

Relative condition number of the matrix exponential in the Frobenius norm.

fractional_matrix_power(A, t)

Compute the fractional power of a matrix.

solve_sylvester(a, b, q)

Computes a solution (X) to the Sylvester equation \(AX + XB = Q\).

solve_continuous_are(a, b, q, r[, e, s, ...])

Solves the continuous-time algebraic Riccati equation (CARE).

solve_discrete_are(a, b, q, r[, e, s, balanced])

Solves the discrete-time algebraic Riccati equation (DARE).

solve_continuous_lyapunov(a, q)

Solves the continuous Lyapunov equation \(AX + XA^H = Q\).

solve_discrete_lyapunov(a, q[, method])

Solves the discrete Lyapunov equation \(AXA^H - X + Q = 0\).

clarkson_woodruff_transform(input_matrix, ...)

Applies a Clarkson-Woodruff Transform/sketch to the input matrix.

Create a block diagonal array from provided arrays.

Construct a circulant matrix.

Create a companion matrix.

convolution_matrix(a, n[, mode])

Construct a convolution matrix.

Discrete Fourier transform matrix.

Returns a symmetric Fiedler matrix

Returns a Fiedler companion matrix

Construct an Hadamard matrix.

Construct a Hankel matrix.

Create an Helmert matrix of order n.

Create a Hilbert matrix of order n.

invhilbert(n[, exact])

Compute the inverse of the Hilbert matrix of order n.

Create a Leslie matrix.

pascal(n[, kind, exact])

Returns the n x n Pascal matrix.

invpascal(n[, kind, exact])

Returns the inverse of the n x n Pascal matrix.

Construct a Toeplitz matrix.

get_blas_funcs(names[, arrays, dtype, ilp64])

Return available BLAS function objects from names.

get_lapack_funcs(names[, arrays, dtype, ilp64])

Return available LAPACK function objects from names.

find_best_blas_type([arrays, dtype])

Find best-matching BLAS/LAPACK type.

scipy.linalg.blas – Low-level BLAS functions

scipy.linalg.lapack – Low-level LAPACK functions

scipy.linalg.cython_blas – Low-level BLAS functions for Cython

scipy.linalg.cython_lapack – Low-level LAPACK functions for Cython

---

## uniform_filter1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html

**Contents:**
- uniform_filter1d#

Calculate a 1-D uniform filter along the given axis.

The lines of the array along the given axis are filtered with a uniform filter of given size.

length of uniform filter

The axis of input along which to calculate. Default is -1.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right.

Filtered array. Has same shape as input.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.ndimage import uniform_filter1d
>>> uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([4, 3, 4, 1, 4, 6, 6, 3])
```

---

## BLAS Functions for Cython#

**URL:** https://docs.scipy.org/doc/scipy/reference/linalg.cython_blas.html

**Contents:**
- BLAS Functions for Cython#

Usable from Cython via:

These wrappers do not check for alignment of arrays. Alignment should be checked before these wrappers are used.

If using cdotu, cdotc, zdotu, zdotc, sladiv, or dladiv, the CYTHON_CCOMPLEX define must be set to 0 during compilation. For example, in a meson.build file when using Meson:

Raw function pointers (Fortran-style pointer arguments):

**Examples:**

Example 1 (unknown):
```unknown
cimport scipy.linalg.cython_blas
```

Example 2 (json):
```json
py.extension_module('ext_module'
    'ext_module.pyx',
    c_args: ['-DCYTHON_CCOMPLEX=0'],
    ...
)
```

---

## expm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html

**Contents:**
- expm#

Compute the matrix exponential of an array.

Input with last two dimensions are square (..., n, n).

The resulting matrix exponential with the same shape of A

Implements the algorithm given in [1], which is essentially a Pade approximation with a variable order that is decided based on the array data.

For input with size n, the memory usage is in the worst case in the order of 8*(n**2). If the input data is not of single and double precision of real and complex dtypes, it is copied to a new array.

For cases n >= 400, the exact 1-norm computation cost, breaks even with 1-norm estimation and from that point on the estimation scheme given in [2] is used to decide on the approximation order.

Awad H. Al-Mohy and Nicholas J. Higham, (2009), “A New Scaling and Squaring Algorithm for the Matrix Exponential”, SIAM J. Matrix Anal. Appl. 31(3):970-989, DOI:10.1137/09074721X

Nicholas J. Higham and Francoise Tisseur (2000), “A Block Algorithm for Matrix 1-Norm Estimation, with an Application to 1-Norm Pseudospectra.” SIAM J. Matrix Anal. Appl. 21(4):1185-1201, DOI:10.1137/S0895479899356080

Matrix version of the formula exp(0) = 1:

Euler’s identity (exp(i*theta) = cos(theta) + i*sin(theta)) applied to a matrix:

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import expm, sinm, cosm
```

Example 2 (json):
```json
>>> expm(np.zeros((3, 2, 2)))
array([[[1., 0.],
        [0., 1.]],

       [[1., 0.],
        [0., 1.]],

       [[1., 0.],
        [0., 1.]]])
```

Example 3 (json):
```json
>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
```

---

## rfft#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft.html

**Contents:**
- rfft#

Compute the 1-D discrete Fourier Transform for real input.

This function computes the 1-D n-point discrete Fourier Transform (DFT) of a real-valued array by means of an efficient algorithm called the Fast Fourier Transform (FFT).

Number of points along transformation axis in the input to use. If n is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros. If n is not given, the length of the input along the axis specified by axis is used.

Axis over which to compute the FFT. If not given, the last axis is used.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified. If n is even, the length of the transformed axis is (n/2)+1. If n is odd, the length is (n+1)/2.

If axis is larger than the last axis of a.

The 1-D FFT of general (complex) input.

The 2-D FFT of real input.

The N-D FFT of real input.

When the DFT is computed for purely real input, the output is Hermitian-symmetric, i.e., the negative frequency terms are just the complex conjugates of the corresponding positive-frequency terms, and the negative-frequency terms are therefore redundant. This function does not compute the negative frequency terms, and the length of the transformed axis of the output is therefore n//2 + 1.

When X = rfft(x) and fs is the sampling frequency, X[0] contains the zero-frequency term 0*fs, which is real due to Hermitian symmetry.

If n is even, A[-1] contains the term representing both positive and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely real. If n is odd, there is no term at fs/2; A[-1] contains the largest positive frequency (fs/2*(n-1)/n), and is complex in the general case.

If the input a contains an imaginary part, it is silently discarded.

Notice how the final element of the fft output is the complex conjugate of the second element, for real input. For rfft, this symmetry is exploited to compute only the non-negative frequency terms.

**Examples:**

Example 1 (python):
```python
>>> import scipy.fft
>>> scipy.fft.fft([0, 1, 0, 0])
array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j]) # may vary
>>> scipy.fft.rfft([0, 1, 0, 0])
array([ 1.+0.j,  0.-1.j, -1.+0.j]) # may vary
```

---

## cosm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cosm.html

**Contents:**
- cosm#

Compute the matrix cosine.

This routine uses expm to compute the matrix exponentials.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Euler’s identity (exp(i*theta) = cos(theta) + i*sin(theta)) applied to a matrix:

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import expm, sinm, cosm
```

Example 2 (json):
```json
>>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
>>> expm(1j*a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
>>> cosm(a) + 1j*sinm(a)
array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
       [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
```

---

## Integration (scipy.integrate)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/integrate.html

**Contents:**
- Integration (scipy.integrate)#
- General integration (quad)#
- General multiple integration (dblquad, tplquad, nquad)#
- Gaussian quadrature#
- Integrating using Samples#
- Faster integration using low-level callback functions#
- Ordinary differential equations (solve_ivp)#
  - Solving a system with a banded Jacobian matrix#
  - References#

The scipy.integrate sub-package provides several integration techniques including an ordinary differential equation integrator. An overview of the module is provided by the help command:

The function quad is provided to integrate a function of one variable between two points. The points can be \(\pm\infty\) (\(\pm\) inf) to indicate infinite limits. For example, suppose you wish to integrate a bessel function jv(2.5, x) along the interval \([0, 4.5].\)

This could be computed using quad:

The first argument to quad is a “callable” Python object (i.e., a function, method, or class instance). Notice the use of a lambda- function in this case as the argument. The next two arguments are the limits of integration. The return value is a tuple, with the first element holding the estimated value of the integral and the second element holding an estimate of the absolute integration error. Notice, that in this case, the true value of this integral is

is the Fresnel sine integral. Note that the numerically-computed integral is within \(1.04\times10^{-11}\) of the exact result — well below the reported error estimate.

If the function to integrate takes additional parameters, they can be provided in the args argument. Suppose that the following integral shall be calculated:

This integral can be evaluated by using the following code:

Infinite inputs are also allowed in quad by using \(\pm\) inf as one of the arguments. For example, suppose that a numerical value for the exponential integral:

is desired (and the fact that this integral can be computed as special.expn(n,x) is forgotten). The functionality of the function special.expn can be replicated by defining a new function vec_expint based on the routine quad:

The function which is integrated can even use the quad argument (though the error bound may underestimate the error due to possible numerical error in the integrand from the use of quad ). The integral in this case is

This last example shows that multiple integration can be handled using repeated calls to quad.

Numerical integration algorithms sample the integrand at a finite number of points. Consequently, they cannot guarantee accurate results (or accuracy estimates) for arbitrary integrands and limits of integration. Consider the Gaussian integral, for example:

Since the integrand is nearly zero except near the origin, we would expect large but finite limits of integration to yield the same result. However:

This happens because the adaptive quadrature routine implemented in quad, while working as designed, does not notice the small, important part of the function within such a large, finite interval. For best results, consider using integration limits that tightly surround the important part of the integrand.

Integrands with several important regions can be broken into pieces as necessary.

The mechanics for double and triple integration have been wrapped up into the functions dblquad and tplquad. These functions take the function to integrate and four, or six arguments, respectively. The limits of all inner integrals need to be defined as functions.

An example of using double integration to compute several values of \(I_{n}\) is shown below:

As example for non-constant limits consider the integral

This integral can be evaluated using the expression below (Note the use of the non-constant lambda functions for the upper limit of the inner integral):

For n-fold integration, scipy provides the function nquad. The integration bounds are an iterable object: either a list of constant bounds, or a list of functions for the non-constant integration bounds. The order of integration (and therefore the bounds) is from the innermost integral to the outermost one.

The integral from above

Note that the order of arguments for f must match the order of the integration bounds; i.e., the inner integral with respect to \(t\) is on the interval \([1, \infty]\) and the outer integral with respect to \(x\) is on the interval \([0, \infty]\).

Non-constant integration bounds can be treated in a similar manner; the example from above

can be evaluated by means of

which is the same result as before.

fixed_quad performs fixed-order Gaussian quadrature over a fixed interval. This function uses the collection of orthogonal polynomials provided by scipy.special, which can calculate the roots and quadrature weights of a large variety of orthogonal polynomials (the polynomials themselves are available as special functions returning instances of the polynomial class — e.g., special.legendre).

If the samples are equally-spaced and the number of samples available is \(2^{k}+1\) for some integer \(k\), then Romberg romb integration can be used to obtain high-precision estimates of the integral using the available samples. Romberg integration uses the trapezoid rule at step-sizes related by a power of two and then performs Richardson extrapolation on these estimates to approximate the integral with a higher degree of accuracy.

In case of arbitrary spaced samples, the two functions trapezoid and simpson are available. They are using Newton-Coates formulas of order 1 and 2 respectively to perform integration. The trapezoidal rule approximates the function as a straight line between adjacent points, while Simpson’s rule approximates the function between three adjacent points as a parabola.

For an odd number of samples that are equally spaced Simpson’s rule is exact if the function is a polynomial of order 3 or less. If the samples are not equally spaced, then the result is exact only if the function is a polynomial of order 2 or less.

This corresponds exactly to

whereas integrating the second function

does not correspond to

because the order of the polynomial in f2 is larger than two.

A user desiring reduced integration times may pass a C function pointer through scipy.LowLevelCallable to quad, dblquad, tplquad or nquad and it will be integrated and return a result in Python. The performance increase here arises from two factors. The primary improvement is faster function evaluation, which is provided by compilation of the function itself. Additionally we have a speedup provided by the removal of function calls between C and Python in quad. This method may provide a speed improvements of ~2x for trivial functions such as sine but can produce a much more noticeable improvements (10x+) for more complex functions. This feature then, is geared towards a user with numerically intensive integrations willing to write a little C to reduce computation time significantly.

The approach can be used, for example, via ctypes in a few simple steps:

1.) Write an integrand function in C with the function signature double f(int n, double *x, void *user_data), where x is an array containing the point the function f is evaluated at, and user_data to arbitrary additional data you want to provide.

2.) Now compile this file to a shared/dynamic library (a quick search will help with this as it is OS-dependent). The user must link any math libraries, etc., used. On linux this looks like:

The output library will be referred to as testlib.so, but it may have a different file extension. A library has now been created that can be loaded into Python with ctypes.

3.) Load shared library into Python using ctypes and set restypes and argtypes - this allows SciPy to interpret the function correctly:

The last void *user_data in the function is optional and can be omitted (both in the C function and ctypes argtypes) if not needed. Note that the coordinates are passed in as an array of doubles rather than a separate argument.

4.) Now integrate the library function as normally, here using nquad:

The Python tuple is returned as expected in a reduced amount of time. All optional parameters can be used with this method including specifying singularities, infinite bounds, etc.

Integrating a set of ordinary differential equations (ODEs) given initial conditions is another useful example. The function solve_ivp is available in SciPy for integrating a first-order vector differential equation:

given initial conditions \(\mathbf{y}\left(0\right)=\mathbf{y}_{0}\), where \(\mathbf{y}\) is a length \(N\) vector and \(\mathbf{f}\) is a mapping from \(\mathbb{R}^{N}\) to \(\mathbb{R}^{N}.\) A higher-order ordinary differential equation can always be reduced to a differential equation of this type by introducing intermediate derivatives into the \(\mathbf{y}\) vector.

For example, suppose it is desired to find the solution to the following second-order differential equation:

with initial conditions \(w\left(0\right)=\frac{1}{\sqrt[3]{3^{2}}\Gamma\left(\frac{2}{3}\right)}\) and \(\left.\frac{dw}{dz}\right|_{z=0}=-\frac{1}{\sqrt[3]{3}\Gamma\left(\frac{1}{3}\right)}.\) It is known that the solution to this differential equation with these boundary conditions is the Airy function

which gives a means to check the integrator using special.airy.

First, convert this ODE into standard form by setting \(\mathbf{y}=\left[\frac{dw}{dz},w\right]\) and \(t=z\). Thus, the differential equation becomes

As an interesting reminder, if \(\mathbf{A}\left(t\right)\) commutes with \(\int_{0}^{t}\mathbf{A}\left(\tau\right)\, d\tau\) under matrix multiplication, then this linear differential equation has an exact solution using the matrix exponential:

However, in this case, \(\mathbf{A}\left(t\right)\) and its integral do not commute.

This differential equation can be solved using the function solve_ivp. It requires the derivative, fprime, the time span [t_start, t_end] and the initial conditions vector, y0, as input arguments and returns an object whose y field is an array with consecutive solution values as columns. The initial conditions are therefore given in the first output column.

As it can be seen solve_ivp determines its time steps automatically if not specified otherwise. To compare the solution of solve_ivp with the airy function the time vector created by solve_ivp is passed to the airy function.

The solution of solve_ivp with its standard parameters shows a big deviation to the airy function. To minimize this deviation, relative and absolute tolerances can be used.

To specify user defined time points for the solution of solve_ivp, solve_ivp offers two possibilities that can also be used complementarily. By passing the t_eval option to the function call solve_ivp returns the solutions of these time points of t_eval in its output.

If the jacobian matrix of function is known, it can be passed to the solve_ivp to achieve better results. Please be aware however that the default integration method RK45 does not support jacobian matrices and thereby another integration method has to be chosen. One of the integration methods that support a jacobian matrix is the for example the Radau method of following example.

odeint can be told that the Jacobian is banded. For a large system of differential equations that are known to be stiff, this can improve performance significantly.

As an example, we’ll solve the 1-D Gray-Scott partial differential equations using the method of lines [MOL]. The Gray-Scott equations for the functions \(u(x, t)\) and \(v(x, t)\) on the interval \(x \in [0, L]\) are

where \(D_u\) and \(D_v\) are the diffusion coefficients of the components \(u\) and \(v\), respectively, and \(f\) and \(k\) are constants. (For more information about the system, see http://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/)

We’ll assume Neumann (i.e., “no flux”) boundary conditions:

To apply the method of lines, we discretize the \(x\) variable by defining the uniformly spaced grid of \(N\) points \(\left\{x_0, x_1, \ldots, x_{N-1}\right\}\), with \(x_0 = 0\) and \(x_{N-1} = L\). We define \(u_j(t) \equiv u(x_k, t)\) and \(v_j(t) \equiv v(x_k, t)\), and replace the \(x\) derivatives with finite differences. That is,

We then have a system of \(2N\) ordinary differential equations:

For convenience, the \((t)\) arguments have been dropped.

To enforce the boundary conditions, we introduce “ghost” points \(x_{-1}\) and \(x_N\), and define \(u_{-1}(t) \equiv u_1(t)\), \(u_N(t) \equiv u_{N-2}(t)\); \(v_{-1}(t)\) and \(v_N(t)\) are defined analogously.

Our complete system of \(2N\) ordinary differential equations is (1) for \(k = 1, 2, \ldots, N-2\), along with (2) and (3).

We can now starting implementing this system in code. We must combine \(\{u_k\}\) and \(\{v_k\}\) into a single vector of length \(2N\). The two obvious choices are \(\{u_0, u_1, \ldots, u_{N-1}, v_0, v_1, \ldots, v_{N-1}\}\) and \(\{u_0, v_0, u_1, v_1, \ldots, u_{N-1}, v_{N-1}\}\). Mathematically, it does not matter, but the choice affects how efficiently odeint can solve the system. The reason is in how the order affects the pattern of the nonzero elements of the Jacobian matrix.

When the variables are ordered as \(\{u_0, u_1, \ldots, u_{N-1}, v_0, v_1, \ldots, v_{N-1}\}\), the pattern of nonzero elements of the Jacobian matrix is

The Jacobian pattern with variables interleaved as \(\{u_0, v_0, u_1, v_1, \ldots, u_{N-1}, v_{N-1}\}\) is

In both cases, there are just five nontrivial diagonals, but when the variables are interleaved, the bandwidth is much smaller. That is, the main diagonal and the two diagonals immediately above and the two immediately below the main diagonal are the nonzero diagonals. This is important, because the inputs mu and ml of odeint are the upper and lower bandwidths of the Jacobian matrix. When the variables are interleaved, mu and ml are 2. When the variables are stacked with \(\{v_k\}\) following \(\{u_k\}\), the upper and lower bandwidths are \(N\).

With that decision made, we can write the function that implements the system of differential equations.

First, we define the functions for the source and reaction terms of the system:

Next, we define the function that computes the right-hand side of the system of differential equations:

We won’t implement a function to compute the Jacobian, but we will tell odeint that the Jacobian matrix is banded. This allows the underlying solver (LSODA) to avoid computing values that it knows are zero. For a large system, this improves the performance significantly, as demonstrated in the following ipython session.

First, we define the required inputs:

Time the computation without taking advantage of the banded structure of the Jacobian matrix:

Now set ml=2 and mu=2, so odeint knows that the Jacobian matrix is banded:

That is quite a bit faster!

Let’s ensure that they have computed the same result:

https://en.wikipedia.org/wiki/Method_of_lines

**Examples:**

Example 1 (julia):
```julia
>>> help(integrate)
 Methods for Integrating Functions given function object.

   quad          -- General purpose integration.
   dblquad       -- General purpose double integration.
   tplquad       -- General purpose triple integration.
   fixed_quad    -- Integrate func(x) using Gaussian quadrature of order n.
   quadrature    -- Integrate with given tolerance using Gaussian quadrature.
   romberg       -- Integrate func using Romberg integration.

 Methods for Integrating Functions given fixed samples.

   trapezoid            -- Use trapezoidal rule to compute integral.
   cumulative_trapezoid -- Use trapezoidal rule to cumulatively compute integral.
   simpson              -- Use Simpson's rule to compute integral from samples.
   romb                 -- Use Romberg Integration to compute integral from
                        -- (2**k + 1) evenly-spaced samples.

   See the special module's orthogonal polynomials (special) for Gaussian
      quadrature roots and weights for other weighting factors and regions.

 Interface to numerical integrators of ODE systems.

   odeint        -- General integration of ordinary differential equations.
   ode           -- Integrate ODE using VODE and ZVODE routines.
```

Example 2 (typescript):
```typescript
>>> import scipy.integrate as integrate
>>> import scipy.special as special
>>> result = integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)
>>> result
(1.1178179380783249, 7.8663172481899801e-09)
```

Example 3 (python):
```python
>>> from numpy import sqrt, sin, cos, pi
>>> I = sqrt(2/pi)*(18.0/27*sqrt(2)*cos(4.5) - 4.0/27*sqrt(2)*sin(4.5) +
...                 sqrt(2*pi) * special.fresnel(3/sqrt(pi))[0])
>>> I
1.117817938088701
```

Example 4 (unknown):
```unknown
>>> print(abs(result[0]-I))
1.03761443881e-11
```

---

## legendre#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre.html

**Contents:**
- legendre#

Defined to be the solution of

\(P_n(x)\) is a polynomial of degree \(n\).

Degree of the polynomial.

If True, scale the leading coefficient to be 1. Default is False.

The polynomials \(P_n\) are orthogonal over \([-1, 1]\) with weight function 1.

Generate the 3rd-order Legendre polynomial 1/2*(5x^3 + 0x^2 - 3x + 0):

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.special import legendre
>>> legendre(3)
poly1d([ 2.5,  0. , -1.5,  0. ])
```

---

## Compressed sparse graph routines (scipy.sparse.csgraph)#

**URL:** https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html

**Contents:**
- Compressed sparse graph routines (scipy.sparse.csgraph)#
- Contents#
- Graph Representations#
  - Directed vs. undirected#

Fast graph algorithms based on sparse matrix representations.

connected_components(csgraph[, directed, ...])

Analyze the connected components of a sparse graph

laplacian(csgraph[, normed, return_diag, ...])

Return the Laplacian of a directed graph.

shortest_path(csgraph[, method, directed, ...])

Perform a shortest-path graph search on a positive directed or undirected graph.

dijkstra(csgraph[, directed, indices, ...])

Dijkstra algorithm using priority queue

floyd_warshall(csgraph[, directed, ...])

Compute the shortest path lengths using the Floyd-Warshall algorithm

bellman_ford(csgraph[, directed, indices, ...])

Compute the shortest path lengths using the Bellman-Ford algorithm.

johnson(csgraph[, directed, indices, ...])

Compute the shortest path lengths using Johnson's algorithm.

yen(csgraph, source, sink, K, *[, directed, ...])

Yen's K-Shortest Paths algorithm on a directed or undirected graph.

breadth_first_order(csgraph, i_start[, ...])

Return a breadth-first ordering starting with specified node.

depth_first_order(csgraph, i_start[, ...])

Return a depth-first ordering starting with specified node.

breadth_first_tree(csgraph, i_start[, directed])

Return the tree generated by a breadth-first search

depth_first_tree(csgraph, i_start[, directed])

Return a tree generated by a depth-first search.

minimum_spanning_tree(csgraph[, overwrite])

Return a minimum spanning tree of an undirected graph

reverse_cuthill_mckee(graph[, symmetric_mode])

Returns the permutation array that orders a sparse CSR or CSC matrix in Reverse-Cuthill McKee ordering.

maximum_flow(csgraph, source, sink)

Maximize the flow between two vertices in a graph.

maximum_bipartite_matching(graph[, perm_type])

Returns a matching of a bipartite graph whose cardinality is at least that of any given matching of the graph.

min_weight_full_bipartite_matching(biadjacency)

Returns the minimum weight full matching of a bipartite graph.

structural_rank(graph)

Compute the structural rank of a graph (matrix) with a given sparsity pattern.

NegativeCycleError([message])

construct_dist_matrix(graph, predecessors[, ...])

Construct distance matrix from a predecessor matrix

csgraph_from_dense(graph[, null_value, ...])

Construct a CSR-format sparse graph from a dense matrix.

csgraph_from_masked(graph)

Construct a CSR-format graph from a masked array.

csgraph_masked_from_dense(graph[, ...])

Construct a masked array graph representation from a dense matrix.

csgraph_to_dense(csgraph[, null_value])

Convert a sparse graph representation to a dense representation

csgraph_to_masked(csgraph)

Convert a sparse graph representation to a masked array representation

reconstruct_path(csgraph, predecessors[, ...])

Construct a tree from a graph and a predecessor list.

This module uses graphs which are stored in a matrix format. A graph with N nodes can be represented by an (N x N) adjacency matrix G. If there is a connection from node i to node j, then G[i, j] = w, where w is the weight of the connection. For nodes i and j which are not connected, the value depends on the representation:

for dense array representations, non-edges are represented by G[i, j] = 0, infinity, or NaN.

for dense masked representations (of type np.ma.MaskedArray), non-edges are represented by masked values. This can be useful when graphs with zero-weight edges are desired.

for sparse array representations, non-edges are represented by non-entries in the matrix. This sort of sparse representation also allows for edges with zero weights.

As a concrete example, imagine that you would like to represent the following undirected graph:

This graph has three nodes, where node 0 and 1 are connected by an edge of weight 2, and nodes 0 and 2 are connected by an edge of weight 1. We can construct the dense, masked, and sparse representations as follows, keeping in mind that an undirected graph is represented by a symmetric matrix:

This becomes more difficult when zero edges are significant. For example, consider the situation when we slightly modify the above graph:

This is identical to the previous graph, except nodes 0 and 2 are connected by an edge of zero weight. In this case, the dense representation above leads to ambiguities: how can non-edges be represented if zero is a meaningful value? In this case, either a masked or sparse representation must be used to eliminate the ambiguity:

Here we have used a utility routine from the csgraph submodule in order to convert the dense representation to a sparse representation which can be understood by the algorithms in submodule. By viewing the data array, we can see that the zero values are explicitly encoded in the graph.

Matrices may represent either directed or undirected graphs. This is specified throughout the csgraph module by a boolean keyword. Graphs are assumed to be directed by default. In a directed graph, traversal from node i to node j can be accomplished over the edge G[i, j], but not the edge G[j, i]. Consider the following dense graph:

When directed=True we get the graph:

In a non-directed graph, traversal from node i to node j can be accomplished over either G[i, j] or G[j, i]. If both edges are not null, and the two have unequal weights, then the smaller of the two is used.

So for the same graph, when directed=False we get the graph:

Note that a symmetric matrix will represent an undirected graph, regardless of whether the ‘directed’ keyword is set to True or False. In this case, using directed=True generally leads to more efficient computation.

The routines in this module accept as input either scipy.sparse representations (csr, csc, or lil format), masked representations, or dense representations with non-edges indicated by zeros, infinities, and NaN entries.

**Examples:**

Example 1 (unknown):
```unknown
G

     (0)
    /   \
   1     2
  /       \
(2)       (1)
```

Example 2 (sql):
```sql
>>> import numpy as np
>>> G_dense = np.array([[0, 2, 1],
...                     [2, 0, 0],
...                     [1, 0, 0]])
>>> G_masked = np.ma.masked_values(G_dense, 0)
>>> from scipy.sparse import csr_array
>>> G_sparse = csr_array(G_dense)
```

Example 3 (unknown):
```unknown
G2

     (0)
    /   \
   0     2
  /       \
(2)       (1)
```

Example 4 (sql):
```sql
>>> import numpy as np
>>> G2_data = np.array([[np.inf, 2,      0     ],
...                     [2,      np.inf, np.inf],
...                     [0,      np.inf, np.inf]])
>>> G2_masked = np.ma.masked_invalid(G2_data)
>>> from scipy.sparse.csgraph import csgraph_from_dense
>>> # G2_sparse = csr_array(G2_data) would give the wrong result
>>> G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)
>>> G2_sparse.data
array([ 2.,  0.,  2.,  0.])
```

---

## mminfo#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mminfo.html

**Contents:**
- mminfo#

Return size and storage parameters from Matrix Market file-like ‘source’.

Matrix Market filename (extension .mtx) or open file-like object

Number of matrix rows.

Number of matrix columns.

Number of non-zero entries of a sparse matrix or rows*cols for a dense matrix.

Either ‘coordinate’ or ‘array’.

Either ‘real’, ‘complex’, ‘pattern’, or ‘integer’.

Either ‘general’, ‘symmetric’, ‘skew-symmetric’, or ‘hermitian’.

Changed in version 1.12.0: C++ implementation.

mminfo(source) returns the number of rows, number of columns, format, field type and symmetry attribute of the source file.

**Examples:**

Example 1 (python):
```python
>>> from io import StringIO
>>> from scipy.io import mminfo
```

Example 2 (unknown):
```unknown
>>> text = '''%%MatrixMarket matrix coordinate real general
...  5 5 7
...  2 3 1.0
...  3 4 2.0
...  3 5 3.0
...  4 1 4.0
...  4 2 5.0
...  4 3 6.0
...  4 4 7.0
... '''
```

Example 3 (unknown):
```unknown
>>> mminfo(StringIO(text))
(5, 5, 7, 'coordinate', 'real', 'general')
```

---

## generic_gradient_magnitude#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_gradient_magnitude.html

**Contents:**
- generic_gradient_magnitude#

Gradient magnitude using a provided gradient function.

Callable with the following signature:

See extra_arguments, extra_keywords below. derivative can assume that input and output are ndarrays. Note that the output from derivative is modified inplace; be careful to copy important inputs before returning them.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

dict of extra keyword arguments to pass to passed function.

Sequence of extra positional arguments to pass to passed function.

The axes over which to apply the filter. If a mode tuple is provided, its length must match the number of axes.

Filtered array. Has the same shape as input.

**Examples:**

Example 1 (unknown):
```unknown
derivative(input, axis, output, mode, cval,
           *extra_arguments, **extra_keywords)
```

---

## fixed_quad#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.fixed_quad.html

**Contents:**
- fixed_quad#

Compute a definite integral using fixed-order Gaussian quadrature.

Integrate func from a to b using Gaussian quadrature of order n.

A Python function or method to integrate (must accept vector inputs). If integrating a vector-valued function, the returned array must have shape (..., len(x)).

Lower limit of integration.

Upper limit of integration.

Extra arguments to pass to function, if any.

Order of quadrature integration. Default is 5.

Gaussian quadrature approximation to the integral

Statically returned value of None

adaptive quadrature using QUADPACK

integrators for sampled data

integrators for sampled data

cumulative integration for sampled data

**Examples:**

Example 1 (python):
```python
>>> from scipy import integrate
>>> import numpy as np
>>> f = lambda x: x**8
>>> integrate.fixed_quad(f, 0.0, 1.0, n=4)
(0.1110884353741496, None)
>>> integrate.fixed_quad(f, 0.0, 1.0, n=5)
(0.11111111111111102, None)
>>> print(1/9.0)  # analytical result
0.1111111111111111
```

Example 2 (rust):
```rust
>>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)
(0.9999999771971152, None)
>>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)
(1.000000000039565, None)
>>> np.sin(np.pi/2)-np.sin(0)  # analytical result
1.0
```

---

## MATLAB® file utilities (scipy.io.matlab)#

**URL:** https://docs.scipy.org/doc/scipy/reference/io.matlab.html

**Contents:**
- MATLAB® file utilities (scipy.io.matlab)#
- Notes#

This submodule is meant to provide lower-level file utilities related to reading and writing MATLAB files.

matfile_version(file_name, *[, appendmat])

Return major, minor tuple depending on apparent mat file type

Exception indicating a read issue.

Warning class for read issues.

Exception indicating a write issue.

Warning class for write issues.

Placeholder for holding read data from structs.

varmats_from_mat(file_obj)

Pull variables out of mat 5 file as a sequence of mat file objects

Subclass of ndarray to signal this is a matlab object.

Subclass for a MATLAB opaque matrix.

Subclass for a MATLAB function.

The following utilities that live in the scipy.io namespace also exist in this namespace:

loadmat(file_name[, mdict, appendmat, spmatrix])

savemat(file_name, mdict[, appendmat, ...])

Save a dictionary of names and arrays into a MATLAB-style .mat file.

whosmat(file_name[, appendmat])

List variables inside a MATLAB file.

MATLAB® is a registered trademark of The MathWorks, Inc., 3 Apple Hill Drive, Natick, MA 01760-2098, USA.

---

## minimum_filter1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter1d.html

**Contents:**
- minimum_filter1d#

Calculate a 1-D minimum filter along the given axis.

The lines of the array along the given axis are filtered with a minimum filter of given size.

length along which to calculate 1D minimum

The axis of input along which to calculate. Default is -1.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right.

Filtered image. Has the same shape as input.

This function implements the MINLIST algorithm [1], as described by Richard Harter [2], and has a guaranteed O(n) performance, n being the input length, regardless of filter size.

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777

http://www.richardhartersworld.com/cri/2001/slidingmin.html

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.ndimage import minimum_filter1d
>>> minimum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([2, 0, 0, 0, 1, 1, 0, 0])
```

---

## sobel#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html

**Contents:**
- sobel#

Calculate a Sobel filter.

The axis of input along which to calculate. Default is -1.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Filtered array. Has the same shape as input.

This function computes the axis-specific Sobel gradient. The horizontal edges can be emphasised with the horizontal transform (axis=0), the vertical edges with the vertical transform (axis=1) and so on for higher dimensions. These can be combined to give the magnitude.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> ascent = datasets.ascent().astype('int32')
>>> sobel_h = ndimage.sobel(ascent, 0)  # horizontal gradient
>>> sobel_v = ndimage.sobel(ascent, 1)  # vertical gradient
>>> magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
>>> magnitude *= 255.0 / np.max(magnitude)  # normalization
>>> fig, axs = plt.subplots(2, 2, figsize=(8, 8))
>>> plt.gray()  # show the filtered result in grayscale
>>> axs[0, 0].imshow(ascent)
>>> axs[0, 1].imshow(sobel_h)
>>> axs[1, 0].imshow(sobel_v)
>>> axs[1, 1].imshow(magnitude)
>>> titles = ["original", "horizontal", "vertical", "magnitude"]
>>> for i, ax in enumerate(axs.ravel()):
...     ax.set_title(titles[i])
...     ax.axis("off")
>>> plt.show()
```

---

## K-means clustering and vector quantization (scipy.cluster.vq)#

**URL:** https://docs.scipy.org/doc/scipy/reference/cluster.vq.html

**Contents:**
- K-means clustering and vector quantization (scipy.cluster.vq)#
- Background information#

Provides routines for k-means clustering, generating code books from k-means models and quantizing vectors by comparing them with centroids in a code book.

whiten(obs[, check_finite])

Normalize a group of observations on a per feature basis.

vq(obs, code_book[, check_finite])

Assign codes from a code book to observations.

kmeans(obs, k_or_guess[, iter, thresh, ...])

Performs k-means on a set of observation vectors forming k clusters.

kmeans2(data, k[, iter, thresh, minit, ...])

Classify a set of observations into k clusters using the k-means algorithm.

The k-means algorithm takes as input the number of clusters to generate, k, and a set of observation vectors to cluster. It returns a set of centroids, one for each of the k clusters. An observation vector is classified with the cluster number or centroid index of the centroid closest to it.

A vector v belongs to cluster i if it is closer to centroid i than any other centroid. If v belongs to i, we say centroid i is the dominating centroid of v. The k-means algorithm tries to minimize distortion, which is defined as the sum of the squared distances between each observation vector and its dominating centroid. The minimization is achieved by iteratively reclassifying the observations into clusters and recalculating the centroids until a configuration is reached in which the centroids are stable. One can also define a maximum number of iterations.

Since vector quantization is a natural application for k-means, information theory terminology is often used. The centroid index or cluster index is also referred to as a “code” and the table mapping codes to centroids and, vice versa, is often referred to as a “code book”. The result of k-means, a set of centroids, can be used to quantize vectors. Quantization aims to find an encoding of vectors that reduces the expected distortion.

All routines expect obs to be an M by N array, where the rows are the observation vectors. The codebook is a k by N array, where the ith row is the centroid of code word i. The observation vectors and centroids have the same feature dimension.

As an example, suppose we wish to compress a 24-bit color image (each pixel is represented by one byte for red, one for blue, and one for green) before sending it over the web. By using a smaller 8-bit encoding, we can reduce the amount of data by two thirds. Ideally, the colors for each of the 256 possible 8-bit encoding values should be chosen to minimize distortion of the color. Running k-means with k=256 generates a code book of 256 codes, which fills up all possible 8-bit sequences. Instead of sending a 3-byte value for each pixel, the 8-bit centroid index (or code word) of the dominating centroid is transmitted. The code book is also sent over the wire so each 8-bit code can be translated back to a 24-bit pixel value representation. If the image of interest was of an ocean, we would expect many 24-bit blues to be represented by 8-bit codes. If it was an image of a human face, more flesh-tone colors would be represented in the code book.

---

## hankel#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hankel.html

**Contents:**
- hankel#

Construct a Hankel matrix.

The Hankel matrix has constant anti-diagonals, with c as its first column and r as its last row. If the first element of r differs from the last element of c, the first element of r is replaced by the last element of c to ensure that anti-diagonals remain constant. If r is not given, then r = zeros_like(c) is assumed.

First column of the matrix. Whatever the actual shape of c, it will be converted to a 1-D array.

Last row of the matrix. If None, r = zeros_like(c) is assumed. r[0] is ignored; the last row of the returned matrix is [c[-1], r[1:]]. Whatever the actual shape of r, it will be converted to a 1-D array.

The Hankel matrix. Dtype is the same as (c[0] + r[0]).dtype.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import hankel
>>> hankel([1, 17, 99])
array([[ 1, 17, 99],
       [17, 99,  0],
       [99,  0,  0]])
>>> hankel([1,2,3,4], [4,7,7,8,9])
array([[1, 2, 3, 4, 7],
       [2, 3, 4, 7, 7],
       [3, 4, 7, 7, 8],
       [4, 7, 7, 8, 9]])
```

---

## Orthogonal distance regression (scipy.odr)#

**URL:** https://docs.scipy.org/doc/scipy/reference/odr.html

**Contents:**
- Orthogonal distance regression (scipy.odr)#
- Package Content#
- Usage information#
  - Introduction#
  - Basic usage#
  - References#

Data(x[, y, we, wd, fix, meta])

RealData(x[, y, sx, sy, covx, covy, fix, meta])

The data, with weightings as actual standard deviations and/or covariances.

Model(fcn[, fjacb, fjacd, extra_args, ...])

The Model class stores information about the function you wish to fit.

ODR(data, model[, beta0, delta0, ifixb, ...])

The ODR class gathers all information and coordinates the running of the main fitting routine.

The Output class stores the output of an ODR run.

odr(fcn, beta0, y, x[, we, wd, fjacb, ...])

Low-level function for ODR.

Warning indicating that the data passed into ODR will cause problems when passed into 'odr' that the user should be aware of.

Exception indicating an error in fitting.

Exception stopping fitting.

Factory function for a general polynomial model.

Arbitrary-dimensional linear model

Univariate linear model

Why Orthogonal Distance Regression (ODR)? Sometimes one has measurement errors in the explanatory (a.k.a., “independent”) variable(s), not just the response (a.k.a., “dependent”) variable(s). Ordinary Least Squares (OLS) fitting procedures treat the data for explanatory variables as fixed, i.e., not subject to error of any kind. Furthermore, OLS procedures require that the response variables be an explicit function of the explanatory variables; sometimes making the equation explicit is impractical and/or introduces errors. ODR can handle both of these cases with ease, and can even reduce to the OLS case if that is sufficient for the problem.

ODRPACK is a FORTRAN-77 library for performing ODR with possibly non-linear fitting functions. It uses a modified trust-region Levenberg-Marquardt-type algorithm [1] to estimate the function parameters. The fitting functions are provided by Python functions operating on NumPy arrays. The required derivatives may be provided by Python functions as well, or may be estimated numerically. ODRPACK can do explicit or implicit ODR fits, or it can do OLS. Input and output variables may be multidimensional. Weights can be provided to account for different variances of the observations, and even covariances between dimensions of the variables.

The scipy.odr package offers an object-oriented interface to ODRPACK, in addition to the low-level odr function.

Additional background information about ODRPACK can be found in the ODRPACK User’s Guide, reading which is recommended.

Define the function you want to fit against.:

Create a Data or RealData instance.:

or, when the actual covariances are known:

Instantiate ODR with your data, model and initial parameter estimate.:

P. T. Boggs and J. E. Rogers, “Orthogonal Distance Regression,” in “Statistical analysis of measurement error models and applications: proceedings of the AMS-IMS-SIAM joint summer research conference held June 10-16, 1989,” Contemporary Mathematics, vol. 112, pg. 186, 1990.

**Examples:**

Example 1 (python):
```python
def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]
```

Example 2 (unknown):
```unknown
linear = Model(f)
```

Example 3 (unknown):
```unknown
mydata = Data(x, y, wd=1./power(sx,2), we=1./power(sy,2))
```

Example 4 (unknown):
```unknown
mydata = RealData(x, y, sx=sx, sy=sy)
```

---

## Sparse linear algebra (scipy.sparse.linalg)#

**URL:** https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

**Contents:**
- Sparse linear algebra (scipy.sparse.linalg)#
- Abstract linear operators#
- Matrix Operations#
- Matrix norms#
- Solving linear problems#
- Matrix factorizations#
- Sparse arrays with structure#
- Exceptions#

LinearOperator(*args, **kwargs)

Common interface for performing matrix vector products

Return A as a LinearOperator.

Compute the inverse of a sparse arrays

Compute the matrix exponential using Pade approximation.

expm_multiply(A, B[, start, stop, num, ...])

Compute the action of the matrix exponential of A on B.

matrix_power(A, power)

Raise a square matrix to the integer power, power.

Norm of a sparse matrix

onenormest(A[, t, itmax, compute_v, compute_w])

Compute a lower bound of the 1-norm of a sparse array.

Direct methods for linear equation systems:

spsolve(A, b[, permc_spec, use_umfpack])

Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

spsolve_triangular(A, b[, lower, ...])

Solve the equation A x = b for x, assuming A is a triangular matrix.

Returns 2-tuple indicating lower/upper triangular structure for sparse A

Return the lower and upper bandwidth of a 2D numeric array.

Return a function for solving a sparse linear system, with A pre-factorized.

Warning for exactly singular matrices.

Select default sparse direct solver to be used.

Iterative methods for linear equation systems:

bicg(A, b[, x0, rtol, atol, maxiter, M, ...])

Solve Ax = b with the BIConjugate Gradient method.

bicgstab(A, b[, x0, rtol, atol, maxiter, M, ...])

Solve Ax = b with the BIConjugate Gradient STABilized method.

cg(A, b[, x0, rtol, atol, maxiter, M, callback])

Solve Ax = b with the Conjugate Gradient method, for a symmetric, positive-definite A.

cgs(A, b[, x0, rtol, atol, maxiter, M, callback])

Solve Ax = b with the Conjugate Gradient Squared method.

gmres(A, b[, x0, rtol, atol, restart, ...])

Solve Ax = b with the Generalized Minimal RESidual method.

lgmres(A, b[, x0, rtol, atol, maxiter, M, ...])

Solve Ax = b with the LGMRES algorithm.

minres(A, b[, x0, rtol, shift, maxiter, M, ...])

Solve Ax = b with the MINimum RESidual method, for a symmetric A.

qmr(A, b[, x0, rtol, atol, maxiter, M1, M2, ...])

Solve Ax = b with the Quasi-Minimal Residual method.

gcrotmk(A, b[, x0, rtol, atol, maxiter, M, ...])

Solve Ax = b with the flexible GCROT(m,k) algorithm.

tfqmr(A, b[, x0, rtol, atol, maxiter, M, ...])

Solve Ax = b with the Transpose-Free Quasi-Minimal Residual method.

Iterative methods for least-squares problems:

lsqr(A, b[, damp, atol, btol, conlim, ...])

Find the least-squares solution to a large, sparse, linear system of equations.

lsmr(A, b[, damp, atol, btol, conlim, ...])

Iterative solver for least-squares problems.

eigs(A[, k, M, sigma, which, v0, ncv, ...])

Find k eigenvalues and eigenvectors of the square matrix A.

eigsh(A[, k, M, sigma, which, v0, ncv, ...])

Find k eigenvalues and eigenvectors of the real symmetric square matrix or complex Hermitian matrix A.

lobpcg(A, X[, B, M, Y, tol, maxiter, ...])

Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).

Singular values problems:

svds(A[, k, ncv, tol, which, v0, maxiter, ...])

Partial singular value decomposition of a sparse matrix.

The svds function supports the following solvers:

Complete or incomplete LU factorizations

splu(A[, permc_spec, diag_pivot_thresh, ...])

Compute the LU decomposition of a sparse, square matrix.

spilu(A[, drop_tol, fill_factor, drop_rule, ...])

Compute an incomplete LU decomposition for a sparse, square matrix.

LU factorization of a sparse matrix.

LaplacianNd(*args, **kwargs)

The grid Laplacian in N dimensions and its eigenvalues/eigenvectors.

ArpackNoConvergence(msg, eigenvalues, ...)

ARPACK iteration did not converge

ArpackError(info[, infodict])

---

## laplace#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html

**Contents:**
- laplace#

N-D Laplace filter based on approximate second derivatives.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

The axes over which to apply the filter. If a mode tuple is provided, its length must match the number of axes.

Filtered array. Has the same shape as input.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = ndimage.laplace(ascent)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## ifftn#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftn.html

**Contents:**
- ifftn#

Compute the N-D inverse discrete Fourier Transform.

This function computes the inverse of the N-D discrete Fourier Transform over any number of axes in an M-D array by means of the Fast Fourier Transform (FFT). In other words, ifftn(fftn(x)) == x to within numerical accuracy.

The input, analogously to ifft, should be ordered in the same way as is returned by fftn, i.e., it should have the term for zero frequency in all axes in the low-order corner, the positive frequency terms in the first half of all axes, the term for the Nyquist frequency in the middle of all axes and the negative frequency terms in the second half of all axes, in order of decreasingly negative frequency.

Input array, can be complex.

Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). This corresponds to n for ifft(x, n). Along any axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros. if s is not given, the shape of the input along the axes specified by axes is used. See notes for issue on ifft zero padding.

Axes over which to compute the IFFT. If not given, the last len(s) axes are used, or all axes if s is also not specified.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axes indicated by axes, or by a combination of s or x, as explained in the parameters section above.

If s and axes have different length.

If an element of axes is larger than the number of axes of x.

The forward N-D FFT, of which ifftn is the inverse.

Undoes fftshift, shifts zero-frequency terms to beginning of array.

Zero-padding, analogously with ifft, is performed by appending zeros to the input along the specified dimension. Although this is the common approach, it might lead to surprising results. If another form of zero padding is desired, it must be performed before ifftn is called.

Create and plot an image with band-limited frequency content:

**Examples:**

Example 1 (json):
```json
>>> import scipy.fft
>>> import numpy as np
>>> x = np.eye(4)
>>> scipy.fft.ifftn(scipy.fft.fftn(x, axes=(0,)), axes=(1,))
array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
       [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
       [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
       [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
```

Example 2 (typescript):
```typescript
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> n = np.zeros((200,200), dtype=complex)
>>> n[60:80, 20:40] = np.exp(1j*rng.uniform(0, 2*np.pi, (20, 20)))
>>> im = scipy.fft.ifftn(n).real
>>> plt.imshow(im)
<matplotlib.image.AxesImage object at 0x...>
>>> plt.show()
```

---

## Multivariate data interpolation on a regular grid (RegularGridInterpolator)#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_regular_grid.html

**Contents:**
- Multivariate data interpolation on a regular grid (RegularGridInterpolator)#
- Batch dimensions of values#
- Uniformly spaced data#

Suppose you have N-dimensional data on a regular grid, and you want to interpolate it. In such a case, RegularGridInterpolator can be useful. Several interpolation strategies are supported: nearest-neighbor, linear, and tensor product splines of odd degree.

Strictly speaking, this class efficiently handles data given on rectilinear grids: hypercubic lattices with possibly unequal spacing between points. The number of points per dimension can be different for different dimensions.

The following example demonstrates its use, and compares the interpolation results using each method.

Suppose we want to interpolate this 2-D function.

Suppose we only know some data on a regular grid.

Creating test points and true values for evaluations.

We can create the interpolator and interpolate test points using each method.

As expected, the higher degree spline interpolations are closest to the true values, though are more expensive to compute than with linear or nearest. The slinear interpolation also matches the linear interpolation.

If your data is such that spline methods produce ringing, you may consider using method="pchip", which uses the tensor product of PCHIP interpolators, a PchipInterpolator per dimension.

If you prefer a functional interface opposed to explicitly creating a class instance, the interpn convenience function offers the equivalent functionality.

Specifically, these two forms give identical results:

For data confined to an (N-1)-dimensional subspace of N-dimensional space, i.e. when one of the grid axes has length 1, the extrapolation along this axis is controlled by the fill_value keyword parameter:

If the input data is such that input dimensions have incommensurate units and differ by many orders of magnitude, the interpolant may have numerical artifacts. Consider rescaling the data before interpolating.

Suppose you have a vector function \(f(x) = y\), where \(x\) and \(y\) are vectors, potentially of different lengths, and you want to sample the function on a grid of \(x\) values. One way to address this is to use the fact that RegularGridInterpolator allows values with trailing dimensions.

In accordance with how 1D interpolators interpret multidimensional arrays, the interpretation is that the first \(N\) dimensions of the values arrays are data dimensions (i.e. they correspond to the points defined by the grid argument), and the trailing dimensions are batch axes. Note that this disagrees with a usual NumPy broadcasting conventions, where broadcasting proceeds along the leading dimensions.

In this example, we evaluated a batch of \(n=5\) functions on a three-dimensional grid. In general, multiple batching dimensions are allowed, and the shape of the result follows by appending the batching shape (in this example, (5,)) to the shape of the input x (in this example, (1,)`).

If you are dealing with data on Cartesian grids with integer coordinates, e.g. resampling image data, these routines may not be the optimal choice. Consider using scipy.ndimage.map_coordinates instead.

For floating-point data on grids with equal spacing, map_coordinates can be easily wrapped into a RegularGridInterpolator look-alike. The following is a bare-bones example originating from the Johanness Buchner’s ‘regulargrid’ package:

This wrapper can be used as a(n almost) drop-in replacement for the RegularGridInterpolator:

Note that the example above uses the map_coordinates boundary conditions. Thus, results of the cubic and quintic interpolations may differ from those of the RegularGridInterpolator. Refer to scipy.ndimage.map_coordinates documentation for more details on boundary conditions and other additional arguments. Finally, we note that this simplified example assumes that the input data is given in the ascending order.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy.interpolate import RegularGridInterpolator
```

Example 2 (python):
```python
>>> def F(u, v):
...     return u * np.cos(u * v) + v * np.sin(u * v)
```

Example 3 (unknown):
```unknown
>>> fit_points = [np.linspace(0, 3, 8), np.linspace(0, 3, 11)]
>>> values = F(*np.meshgrid(*fit_points, indexing='ij'))
```

Example 4 (unknown):
```unknown
>>> ut, vt = np.meshgrid(np.linspace(0, 3, 80), np.linspace(0, 3, 80), indexing='ij')
>>> true_values = F(ut, vt)
>>> test_points = np.array([ut.ravel(), vt.ravel()]).T
```

---

## spline_filter1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filter1d.html

**Contents:**
- spline_filter1d#

Calculate a 1-D spline filter along the given axis.

The lines of the array along the given axis are filtered by a spline filter. The order of the spline must be >= 2 and <= 5.

The order of the spline, default is 3.

The axis along which the spline filter is applied. Default is the last axis.

The array in which to place the output, or the dtype of the returned array. Default is numpy.float64.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘mirror’. Behavior for each valid value is as follows (see additional plots and details on boundary modes):

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

This is a synonym for ‘reflect’.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. No interpolation is performed beyond the edges of the input.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. Interpolation occurs for samples outside the input’s extent as well.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

The input is extended by wrapping around to the opposite edge, but in a way such that the last point and initial point exactly overlap. In this case it is not well defined which sample will be chosen at the point of overlap.

Multidimensional spline filter.

All of the interpolation functions in ndimage do spline interpolation of the input image. If using B-splines of order > 1, the input image values have to be converted to B-spline coefficients first, which is done by applying this 1-D filter sequentially along all axes of the input. All functions that require B-spline coefficients will automatically filter their inputs, a behavior controllable with the prefilter keyword argument. For functions that accept a mode parameter, the result will only be correct if it matches the mode used when filtering.

For complex-valued input, this function processes the real and imaginary components independently.

Added in version 1.6.0: Complex-valued support added.

We can filter an image using 1-D spline along the given axis:

**Examples:**

Example 1 (python):
```python
>>> from scipy.ndimage import spline_filter1d
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> orig_img = np.eye(20)  # create an image
>>> orig_img[10, :] = 1.0
>>> sp_filter_axis_0 = spline_filter1d(orig_img, axis=0)
>>> sp_filter_axis_1 = spline_filter1d(orig_img, axis=1)
>>> f, ax = plt.subplots(1, 3, sharex=True)
>>> for ind, data in enumerate([[orig_img, "original image"],
...             [sp_filter_axis_0, "spline filter (axis=0)"],
...             [sp_filter_axis_1, "spline filter (axis=1)"]]):
...     ax[ind].imshow(data[0], cmap='gray_r')
...     ax[ind].set_title(data[1])
>>> plt.tight_layout()
>>> plt.show()
```

---

## uniform_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html

**Contents:**
- uniform_filter#

Multidimensional uniform filter.

The sizes of the uniform filter are given for each axis as a sequence, or as a single number, in which case the size is equal for all axes.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for size, origin, and/or mode must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

Filtered array. Has the same shape as input.

The multidimensional filter is implemented as a sequence of 1-D uniform filters. The intermediate arrays are stored in the same data type as the output. Therefore, for output types with a limited precision, the results may be imprecise because intermediate results may be stored with insufficient precision.

The behavior of this function with NaN elements is undefined. To control behavior in the presence of NaNs, consider using vectorized_filter.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = ndimage.uniform_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## pascal#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pascal.html

**Contents:**
- pascal#

Returns the n x n Pascal matrix.

The Pascal matrix is a matrix containing the binomial coefficients as its elements.

The size of the matrix to create; that is, the result is an n x n matrix.

Must be one of ‘symmetric’, ‘lower’, or ‘upper’. Default is ‘symmetric’.

If exact is True, the result is either an array of type numpy.uint64 (if n < 35) or an object array of Python long integers. If exact is False, the coefficients in the matrix are computed using scipy.special.comb with exact=False. The result will be a floating point array, and the values in the array will not be the exact coefficients, but this version is much faster than exact=True.

See https://en.wikipedia.org/wiki/Pascal_matrix for more information about Pascal matrices.

Added in version 0.11.0.

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.linalg import pascal
>>> pascal(4)
array([[ 1,  1,  1,  1],
       [ 1,  2,  3,  4],
       [ 1,  3,  6, 10],
       [ 1,  4, 10, 20]], dtype=uint64)
>>> pascal(4, kind='lower')
array([[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 2, 1, 0],
       [1, 3, 3, 1]], dtype=uint64)
>>> pascal(50)[-1, -1]
25477612258980856902730428600
>>> from scipy.special import comb
>>> comb(98, 49, exact=True)
25477612258980856902730428600
```

---

## readsav#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.readsav.html

**Contents:**
- readsav#

Read an IDL .sav file.

Name of the IDL save file.

Dictionary in which to insert .sav file variables.

By default, the object return is not a Python dictionary, but a case-insensitive dictionary with item, attribute, and call access to variables. To get a standard Python dictionary, set this option to True.

This option only has an effect for .sav files written with the /compress option. If a file name is specified, compressed .sav files are uncompressed to this file. Otherwise, readsav will use the tempfile module to determine a temporary filename automatically, and will remove the temporary file upon successfully reading it in.

Whether to print out information about the save file, including the records read, and available variables.

If python_dict is set to False (default), this function returns a case-insensitive dictionary with item, attribute, and call access to variables. If python_dict is set to True, this function returns a Python dictionary with all variable names in lowercase. If idict was specified, then variables are written to the dictionary specified, and the updated dictionary is returned.

Get the filename for an example .sav file from the tests/data directory.

Load the .sav file contents.

Get keys of the .sav file contents.

Access a content with a key.

**Examples:**

Example 1 (sql):
```sql
>>> from os.path import dirname, join as pjoin
>>> import scipy.io as sio
>>> from scipy.io import readsav
```

Example 2 (unknown):
```unknown
>>> data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
>>> sav_fname = pjoin(data_dir, 'array_float32_1d.sav')
```

Example 3 (unknown):
```unknown
>>> sav_data = readsav(sav_fname)
```

Example 4 (unknown):
```unknown
>>> print(sav_data.keys())
dict_keys(['array1d'])
```

---

## LinearNDInterpolator#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html

**Contents:**
- LinearNDInterpolator#

Piecewise linear interpolator in N > 1 dimensions.

Added in version 0.9.

2-D array of data point coordinates, or a precomputed Delaunay triangulation.

N-D array of data values at points. The length of values along the first axis must be equal to the length of points. Unlike some interpolators, the interpolation axis cannot be changed.

Value used to fill in for requested points outside of the convex hull of the input points. If not provided, then the default is nan.

Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.

Evaluate interpolator at given points.

Interpolate unstructured D-D data.

Nearest-neighbor interpolator in N dimensions.

Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.

Interpolation on a regular grid or rectilinear grid.

Interpolator on a regular or rectilinear grid in arbitrary dimensions (interpn wraps this class).

The interpolant is constructed by triangulating the input data with Qhull [1], and on each triangle performing linear barycentric interpolation.

For data on a regular grid use interpn instead.

http://www.qhull.org/

We can interpolate values on a 2D plane:

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.interpolate import LinearNDInterpolator
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> x = rng.random(10) - 0.5
>>> y = rng.random(10) - 0.5
>>> z = np.hypot(x, y)
>>> X = np.linspace(min(x), max(x))
>>> Y = np.linspace(min(y), max(y))
>>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
>>> interp = LinearNDInterpolator(list(zip(x, y)), z)
>>> Z = interp(X, Y)
>>> plt.pcolormesh(X, Y, Z, shading='auto')
>>> plt.plot(x, y, "ok", label="input point")
>>> plt.legend()
>>> plt.colorbar()
>>> plt.axis("equal")
>>> plt.show()
```

---

## loadmat#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

**Contents:**
- loadmat#

Name of the mat file (do not need .mat extension if appendmat==True). Can also pass open file-like object.

Dictionary in which to insert matfile variables.

True to append the .mat extension to the end of the given filename, if not already present. Default is True.

If True, return sparse coo_matrix. Otherwise return coo_array. Only relevant for sparse variables.

None by default, implying byte order guessed from mat file. Otherwise can be one of (‘native’, ‘=’, ‘little’, ‘<’, ‘BIG’, ‘>’).

If True, return arrays in same dtype as would be loaded into MATLAB (instead of the dtype with which they are saved).

Whether to squeeze unit matrix dimensions or not.

Whether to convert char arrays to string arrays.

Returns matrices as would be loaded by MATLAB (implies squeeze_me=False, chars_as_strings=False, mat_dtype=True, struct_as_record=True).

Whether to load MATLAB structs as NumPy record arrays, or as old-style NumPy arrays with dtype=object. Setting this flag to False replicates the behavior of scipy version 0.7.x (returning NumPy object arrays). The default setting is True, because it allows easier round-trip load and save of MATLAB files.

Whether the length of compressed sequences in the MATLAB file should be checked, to ensure that they are not longer than we expect. It is advisable to enable this (the default) because overlong compressed sequences in MATLAB files generally indicate that the files have experienced some sort of corruption.

If None (the default) - read all variables in file. Otherwise, variable_names should be a sequence of strings, giving names of the MATLAB variables to read from the file. The reader will skip any variable with a name not in this sequence, possibly saving some read processing.

If True, return a simplified dict structure (which is useful if the mat file contains cell arrays). Note that this only affects the structure of the result and not its contents (which is identical for both output structures). If True, this automatically sets struct_as_record to False and squeeze_me to True, which is required to simplify cells.

The codec to use for decoding characters, which are stored as uint16 values. The default uses the system encoding, but this can be manually set to other values such as ‘ascii’, ‘latin1’, and ‘utf-8’. This parameter is relevant only for files stored as v6 and above, and not for files stored as v4.

dictionary with variable names as keys, and loaded matrices as values.

v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

You will need an HDF5 Python library to read MATLAB 7.3 format mat files. Because SciPy does not supply one, we do not implement the HDF5 / 7.3 interface here.

Get the filename for an example .mat file from the tests/data directory.

Load the .mat file contents.

The result is a dictionary, one key/value pair for each variable:

By default SciPy reads MATLAB structs as structured NumPy arrays where the dtype fields are of type object and the names correspond to the MATLAB struct field names. This can be disabled by setting the optional argument struct_as_record=False.

Get the filename for an example .mat file that contains a MATLAB struct called teststruct and load the contents.

The size of the structured array is the size of the MATLAB struct, not the number of elements in any particular field. The shape defaults to 2-D unless the optional argument squeeze_me=True, in which case all length 1 dimensions are removed.

Get the ‘stringfield’ of the first element in the MATLAB struct.

Get the first element of the ‘doublefield’.

Load the MATLAB struct, squeezing out length 1 dimensions, and get the item from the ‘complexfield’.

**Examples:**

Example 1 (sql):
```sql
>>> from os.path import dirname, join as pjoin
>>> import scipy.io as sio
```

Example 2 (unknown):
```unknown
>>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
>>> mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')
```

Example 3 (unknown):
```unknown
>>> mat_contents = sio.loadmat(mat_fname, spmatrix=False)
```

Example 4 (json):
```json
>>> sorted(mat_contents.keys())
['__globals__', '__header__', '__version__', 'testdouble']
>>> mat_contents['testdouble']
array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
        3.92699082, 4.71238898, 5.49778714, 6.28318531]])
```

---

## Statistical functions (scipy.stats)#

**URL:** https://docs.scipy.org/doc/scipy/reference/stats.html

**Contents:**
- Statistical functions (scipy.stats)#
- Probability distributions#
  - Continuous distributions#
  - Multivariate distributions#
  - Discrete distributions#
- Summary statistics#
- Frequency statistics#
- Hypothesis Tests and related functions#
  - One Sample Tests / Paired Sample Tests#
  - Association/Correlation Tests#

This module contains a large number of probability distributions, summary and frequency statistics, correlation functions and statistical tests, masked statistics, kernel density estimation, quasi-Monte Carlo functionality, and more.

Statistics is a very large area, and there are topics that are out of scope for SciPy and are covered by other packages. Some of the most important ones are:

statsmodels: regression, linear models, time series analysis, extensions to topics also covered by scipy.stats.

Pandas: tabular data, time series functionality, interfaces to other statistical languages.

PyMC: Bayesian statistical modeling, probabilistic machine learning.

scikit-learn: classification, regression, model selection.

Seaborn: statistical data visualization.

rpy2: Python to R bridge.

Each univariate distribution is an instance of a subclass of rv_continuous (rv_discrete for discrete distributions):

rv_continuous([momtype, a, b, xtol, ...])

A generic continuous random variable class meant for subclassing.

rv_discrete([a, b, name, badvalue, ...])

A generic discrete random variable class meant for subclassing.

rv_histogram(histogram, *args[, density])

Generates a distribution given by a histogram.

An alpha continuous random variable.

An anglit continuous random variable.

An arcsine continuous random variable.

A beta continuous random variable.

A beta prime continuous random variable.

A Bradford continuous random variable.

A Burr (Type III) continuous random variable.

A Burr (Type XII) continuous random variable.

A Cauchy continuous random variable.

A chi continuous random variable.

A chi-squared continuous random variable.

A cosine continuous random variable.

Crystalball distribution

A double gamma continuous random variable.

A double Pareto lognormal continuous random variable.

A double Weibull continuous random variable.

An Erlang continuous random variable.

An exponential continuous random variable.

An exponentially modified Normal continuous random variable.

An exponentiated Weibull continuous random variable.

An exponential power continuous random variable.

An F continuous random variable.

A fatigue-life (Birnbaum-Saunders) continuous random variable.

A Fisk continuous random variable.

A folded Cauchy continuous random variable.

A folded normal continuous random variable.

A generalized logistic continuous random variable.

A generalized normal continuous random variable.

A generalized Pareto continuous random variable.

A generalized exponential continuous random variable.

A generalized extreme value continuous random variable.

A Gauss hypergeometric continuous random variable.

A gamma continuous random variable.

A generalized gamma continuous random variable.

A generalized half-logistic continuous random variable.

A generalized hyperbolic continuous random variable.

A Generalized Inverse Gaussian continuous random variable.

A Gibrat continuous random variable.

A Gompertz (or truncated Gumbel) continuous random variable.

A right-skewed Gumbel continuous random variable.

A left-skewed Gumbel continuous random variable.

A Half-Cauchy continuous random variable.

A half-logistic continuous random variable.

A half-normal continuous random variable.

The upper half of a generalized normal continuous random variable.

A hyperbolic secant continuous random variable.

An inverted gamma continuous random variable.

An inverse Gaussian continuous random variable.

An inverted Weibull continuous random variable.

An Irwin-Hall (Uniform Sum) continuous random variable.

Jones and Faddy skew-t distribution.

A Johnson SB continuous random variable.

A Johnson SU continuous random variable.

Kappa 4 parameter distribution.

Kappa 3 parameter distribution.

Kolmogorov-Smirnov one-sided test statistic distribution.

Kolmogorov-Smirnov two-sided test statistic distribution.

Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic.

A Landau continuous random variable.

A Laplace continuous random variable.

An asymmetric Laplace continuous random variable.

A Levy continuous random variable.

A left-skewed Levy continuous random variable.

A Levy-stable continuous random variable.

A logistic (or Sech-squared) continuous random variable.

A log gamma continuous random variable.

A log-Laplace continuous random variable.

A lognormal continuous random variable.

A loguniform or reciprocal continuous random variable.

A Lomax (Pareto of the second kind) continuous random variable.

A Maxwell continuous random variable.

A Mielke Beta-Kappa / Dagum continuous random variable.

A Moyal continuous random variable.

A Nakagami continuous random variable.

A non-central chi-squared continuous random variable.

A non-central F distribution continuous random variable.

A non-central Student's t continuous random variable.

A normal continuous random variable.

A Normal Inverse Gaussian continuous random variable.

A Pareto continuous random variable.

A pearson type III continuous random variable.

A power-function continuous random variable.

A power log-normal continuous random variable.

A power normal continuous random variable.

An R-distributed (symmetric beta) continuous random variable.

A Rayleigh continuous random variable.

A relativistic Breit-Wigner random variable.

A Rice continuous random variable.

A reciprocal inverse Gaussian continuous random variable.

A semicircular continuous random variable.

A skewed Cauchy random variable.

A skew-normal random variable.

A studentized range continuous random variable.

A Student's t continuous random variable.

A trapezoidal continuous random variable.

A triangular continuous random variable.

A truncated exponential continuous random variable.

A truncated normal continuous random variable.

An upper truncated Pareto continuous random variable.

A doubly truncated Weibull minimum continuous random variable.

A Tukey-Lamdba continuous random variable.

A uniform continuous random variable.

A Von Mises continuous random variable.

A Von Mises continuous random variable.

A Wald continuous random variable.

Weibull minimum continuous random variable.

Weibull maximum continuous random variable.

A wrapped Cauchy continuous random variable.

The fit method of the univariate continuous distributions uses maximum likelihood estimation to fit the distribution to a data set. The fit method can accept regular data or censored data. Censored data is represented with instances of the CensoredData class.

CensoredData([uncensored, left, right, interval])

Instances of this class represent censored data.

A multivariate normal random variable.

A matrix normal random variable.

A Dirichlet random variable.

dirichlet_multinomial

A Dirichlet multinomial random variable.

A Wishart random variable.

An inverse Wishart random variable.

A multinomial random variable.

A Special Orthogonal matrix (SO(N)) random variable.

An Orthogonal matrix (O(N)) random variable.

A matrix-valued U(N) random variable.

A random correlation matrix.

A multivariate t-distributed random variable.

multivariate_hypergeom

A multivariate hypergeometric random variable.

Normal-inverse-gamma distribution.

Contingency tables from independent samples with fixed marginal sums.

A vector-valued uniform direction.

A von Mises-Fisher variable.

scipy.stats.multivariate_normal methods accept instances of the following class to represent the covariance.

Representation of a covariance matrix

A Bernoulli discrete random variable.

A beta-binomial discrete random variable.

A beta-negative-binomial discrete random variable.

A binomial discrete random variable.

A Boltzmann (Truncated Discrete Exponential) random variable.

A Laplacian discrete random variable.

A geometric discrete random variable.

A hypergeometric discrete random variable.

A Logarithmic (Log-Series, Series) discrete random variable.

A negative binomial discrete random variable.

A Fisher's noncentral hypergeometric discrete random variable.

nchypergeom_wallenius

A Wallenius' noncentral hypergeometric discrete random variable.

A negative hypergeometric discrete random variable.

A Planck discrete exponential random variable.

A Poisson discrete random variable.

A Poisson Binomial discrete random variable.

A uniform discrete random variable.

A Skellam discrete random variable.

A Yule-Simon discrete random variable.

A Zipf (Zeta) discrete random variable.

A Zipfian discrete random variable.

An overview of statistical functions is given below. Many of these functions have a similar version in scipy.stats.mstats which work for masked arrays.

describe(a[, axis, ddof, bias, nan_policy])

Compute several descriptive statistics of the passed array.

gmean(a[, axis, dtype, weights, nan_policy, ...])

Compute the weighted geometric mean along the specified axis.

hmean(a[, axis, dtype, weights, nan_policy, ...])

Calculate the weighted harmonic mean along the specified axis.

pmean(a, p, *[, axis, dtype, weights, ...])

Calculate the weighted power mean along the specified axis.

kurtosis(a[, axis, fisher, bias, ...])

Compute the kurtosis (Fisher or Pearson) of a dataset.

mode(a[, axis, nan_policy, keepdims])

Return an array of the modal (most common) value in the passed array.

moment(a[, order, axis, nan_policy, center, ...])

Calculate the nth moment about the mean for a sample.

lmoment(sample[, order, axis, sorted, ...])

Compute L-moments of a sample from a continuous distribution

expectile(a[, alpha, weights])

Compute the expectile at the specified level.

skew(a[, axis, bias, nan_policy, keepdims])

Compute the sample skewness of a data set.

kstat(data[, n, axis, nan_policy, keepdims])

Return the n th k-statistic ( 1<=n<=4 so far).

kstatvar(data[, n, axis, nan_policy, keepdims])

Return an unbiased estimator of the variance of the k-statistic.

tmean(a[, limits, inclusive, axis, ...])

Compute the trimmed mean.

tvar(a[, limits, inclusive, axis, ddof, ...])

Compute the trimmed variance.

tmin(a[, lowerlimit, axis, inclusive, ...])

Compute the trimmed minimum.

tmax(a[, upperlimit, axis, inclusive, ...])

Compute the trimmed maximum.

tstd(a[, limits, inclusive, axis, ddof, ...])

Compute the trimmed sample standard deviation.

tsem(a[, limits, inclusive, axis, ddof, ...])

Compute the trimmed standard error of the mean.

variation(a[, axis, nan_policy, ddof, keepdims])

Compute the coefficient of variation.

Find repeats and repeat counts.

rankdata(a[, method, axis, nan_policy])

Assign ranks to data, dealing with ties appropriately.

Tie correction factor for Mann-Whitney U and Kruskal-Wallis H tests.

trim_mean(a, proportiontocut[, axis])

Return mean of array after trimming a specified fraction of extreme values

gstd(a[, axis, ddof, keepdims, nan_policy])

Calculate the geometric standard deviation of an array.

iqr(x[, axis, rng, scale, nan_policy, ...])

Compute the interquartile range of the data along the specified axis.

sem(a[, axis, ddof, nan_policy, keepdims])

Compute standard error of the mean.

bayes_mvs(data[, alpha])

Bayesian confidence intervals for the mean, var, and std.

'Frozen' distributions for mean, variance, and standard deviation of data.

entropy(pk[, qk, base, axis, nan_policy, ...])

Calculate the Shannon entropy/relative entropy of given distribution(s).

differential_entropy(values, *[, ...])

Given a sample of a distribution, estimate the differential entropy.

median_abs_deviation(x[, axis, center, ...])

Compute the median absolute deviation of the data along the given axis.

cumfreq(a[, numbins, defaultreallimits, weights])

Return a cumulative frequency histogram, using the histogram function.

quantile(x, p, *[, method, axis, ...])

Compute the p-th quantile of the data along the specified axis.

percentileofscore(a, score[, kind, nan_policy])

Compute the percentile rank of a score relative to a list of scores.

scoreatpercentile(a, per[, limit, ...])

Calculate the score at a given percentile of the input sequence.

relfreq(a[, numbins, defaultreallimits, weights])

Return a relative frequency histogram, using the histogram function.

binned_statistic(x, values[, statistic, ...])

Compute a binned statistic for one or more sets of data.

binned_statistic_2d(x, y, values[, ...])

Compute a bidimensional binned statistic for one or more sets of data.

binned_statistic_dd(sample, values[, ...])

Compute a multidimensional binned statistic for a set of data.

SciPy has many functions for performing hypothesis tests that return a test statistic and a p-value, and several of them return confidence intervals and/or other related information.

The headings below are based on common uses of the functions within, but due to the wide variety of statistical procedures, any attempt at coarse-grained categorization will be imperfect. Also, note that tests within the same heading are not interchangeable in general (e.g. many have different distributional assumptions).

One sample tests are typically used to assess whether a single sample was drawn from a specified distribution or a distribution with specified properties (e.g. zero mean).

ttest_1samp(a, popmean[, axis, nan_policy, ...])

Calculate the T-test for the mean of ONE group of scores.

binomtest(k, n[, p, alternative])

Perform a test that the probability of success is p.

quantile_test(x, *[, q, p, alternative])

Perform a quantile test and compute a confidence interval of the quantile.

skewtest(a[, axis, nan_policy, alternative, ...])

Test whether the skew is different from the normal distribution.

kurtosistest(a[, axis, nan_policy, ...])

Test whether a dataset has normal kurtosis.

normaltest(a[, axis, nan_policy, keepdims])

Test whether a sample differs from a normal distribution.

jarque_bera(x, *[, axis, nan_policy, keepdims])

Perform the Jarque-Bera goodness of fit test on sample data.

shapiro(x, *[, axis, nan_policy, keepdims])

Perform the Shapiro-Wilk test for normality.

Anderson-Darling test for data coming from a particular distribution.

cramervonmises(rvs, cdf[, args, axis, ...])

Perform the one-sample Cramér-von Mises test for goodness of fit.

ks_1samp(x, cdf[, args, alternative, ...])

Performs the one-sample Kolmogorov-Smirnov test for goodness of fit.

goodness_of_fit(dist, data, *[, ...])

Perform a goodness of fit test comparing data to a distribution family.

chisquare(f_obs[, f_exp, ddof, axis, ...])

Perform Pearson's chi-squared test.

power_divergence(f_obs[, f_exp, ddof, axis, ...])

Cressie-Read power divergence statistic and goodness of fit test.

Paired sample tests are often used to assess whether two samples were drawn from the same distribution; they differ from the independent sample tests below in that each observation in one sample is treated as paired with a closely-related observation in the other sample (e.g. when environmental factors are controlled between observations within a pair but not among pairs). They can also be interpreted or used as one-sample tests (e.g. tests on the mean or median of differences between paired observations).

ttest_rel(a, b[, axis, nan_policy, ...])

Calculate the t-test on TWO RELATED samples of scores, a and b.

wilcoxon(x[, y, zero_method, correction, ...])

Calculate the Wilcoxon signed-rank test.

These tests are often used to assess whether there is a relationship (e.g. linear) between paired observations in multiple samples or among the coordinates of multivariate observations.

linregress(x, y[, alternative, axis, ...])

Calculate a linear least-squares regression for two sets of measurements.

pearsonr(x, y, *[, alternative, method, axis])

Pearson correlation coefficient and p-value for testing non-correlation.

spearmanr(a[, b, axis, nan_policy, alternative])

Calculate a Spearman correlation coefficient with associated p-value.

pointbiserialr(x, y, *[, axis, nan_policy, ...])

Calculate a point biserial correlation coefficient and its p-value.

kendalltau(x, y, *[, nan_policy, method, ...])

Calculate Kendall's tau, a correlation measure for ordinal data.

chatterjeexi(x, y, *[, axis, y_continuous, ...])

Compute the xi correlation and perform a test of independence

weightedtau(x, y[, rank, weigher, additive, ...])

Compute a weighted version of Kendall's \(\tau\).

somersd(x[, y, alternative])

Calculates Somers' D, an asymmetric measure of ordinal association.

siegelslopes(y[, x, method, axis, ...])

Computes the Siegel estimator for a set of points (x, y).

theilslopes(y[, x, alpha, method, axis, ...])

Computes the Theil-Sen estimator for a set of points (x, y).

page_trend_test(data[, ranked, ...])

Perform Page's Test, a measure of trend in observations between treatments.

multiscale_graphcorr(x, y[, ...])

Computes the Multiscale Graph Correlation (MGC) test statistic.

These association tests and are to work with samples in the form of contingency tables. Supporting functions are available in scipy.stats.contingency.

chi2_contingency(observed[, correction, ...])

Chi-square test of independence of variables in a contingency table.

fisher_exact(table[, alternative, method])

Perform a Fisher exact test on a contingency table.

barnard_exact(table[, alternative, pooled, n])

Perform a Barnard exact test on a 2x2 contingency table.

boschloo_exact(table[, alternative, n])

Perform Boschloo's exact test on a 2x2 contingency table.

Independent sample tests are typically used to assess whether multiple samples were independently drawn from the same distribution or different distributions with a shared property (e.g. equal means).

Some tests are specifically for comparing two samples.

ttest_ind_from_stats(mean1, std1, nobs1, ...)

T-test for means of two independent samples from descriptive statistics.

poisson_means_test(k1, n1, k2, n2, *[, ...])

Performs the Poisson means test, AKA the "E-test".

ttest_ind(a, b, *[, axis, equal_var, ...])

Calculate the T-test for the means of two independent samples of scores.

mannwhitneyu(x, y[, use_continuity, ...])

Perform the Mann-Whitney U rank test on two independent samples.

bws_test(x, y, *[, alternative, method])

Perform the Baumgartner-Weiss-Schindler test on two independent samples.

ranksums(x, y[, alternative, axis, ...])

Compute the Wilcoxon rank-sum statistic for two samples.

brunnermunzel(x, y[, alternative, ...])

Compute the Brunner-Munzel test on samples x and y.

mood(x, y[, axis, alternative, nan_policy, ...])

Perform Mood's test for equal scale parameters.

ansari(x, y[, alternative, axis, ...])

Perform the Ansari-Bradley test for equal scale parameters.

cramervonmises_2samp(x, y[, method, axis, ...])

Perform the two-sample Cramér-von Mises test for goodness of fit.

epps_singleton_2samp(x, y[, t, axis, ...])

Compute the Epps-Singleton (ES) test statistic.

ks_2samp(data1, data2[, alternative, ...])

Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.

kstest(rvs, cdf[, args, N, alternative, ...])

Performs the (one-sample or two-sample) Kolmogorov-Smirnov test for goodness of fit.

Others are generalized to multiple samples.

f_oneway(*samples[, axis, equal_var, ...])

Perform one-way ANOVA.

tukey_hsd(*args[, equal_var])

Perform Tukey's HSD test for equality of means over multiple treatments.

dunnett(*samples, control[, alternative, ...])

Dunnett's test: multiple comparisons of means against a control group.

kruskal(*samples[, nan_policy, axis, keepdims])

Compute the Kruskal-Wallis H-test for independent samples.

alexandergovern(*samples[, nan_policy, ...])

Performs the Alexander Govern test.

fligner(*samples[, center, proportiontocut, ...])

Perform Fligner-Killeen test for equality of variance.

levene(*samples[, center, proportiontocut, ...])

Perform Levene test for equal variances.

bartlett(*samples[, axis, nan_policy, keepdims])

Perform Bartlett's test for equal variances.

median_test(*samples[, ties, correction, ...])

Perform a Mood's median test.

friedmanchisquare(*samples[, axis, ...])

Compute the Friedman test for repeated samples.

anderson_ksamp(samples[, midrank, method])

The Anderson-Darling test for k-samples.

The following functions can reproduce the p-value and confidence interval results of most of the functions above, and often produce accurate results in a wider variety of conditions. They can also be used to perform hypothesis tests and generate confidence intervals for custom statistics. This flexibility comes at the cost of greater computational requirements and stochastic results.

monte_carlo_test(data, rvs, statistic, *[, ...])

Perform a Monte Carlo hypothesis test.

permutation_test(data, statistic, *[, ...])

Performs a permutation test of a given statistic on provided data.

bootstrap(data, statistic, *[, n_resamples, ...])

Compute a two-sided bootstrap confidence interval of a statistic.

power(test, rvs, n_observations, *[, ...])

Simulate the power of a hypothesis test under an alternative hypothesis.

Instances of the following object can be passed into some hypothesis test functions to perform a resampling or Monte Carlo version of the hypothesis test.

MonteCarloMethod([n_resamples, batch, rvs, rng])

Configuration information for a Monte Carlo hypothesis test.

PermutationMethod([n_resamples, batch, ...])

Configuration information for a permutation hypothesis test.

BootstrapMethod([n_resamples, batch, ...])

Configuration information for a bootstrap confidence interval.

These functions are for assessing the results of individual tests as a whole. Functions for performing specific multiple hypothesis tests (e.g. post hoc tests) are listed above.

combine_pvalues(pvalues[, method, weights, ...])

Combine p-values from independent tests that bear upon the same hypothesis.

false_discovery_control(ps, *[, axis, method])

Adjust p-values to control the false discovery rate.

The following functions are related to the tests above but do not belong in the above categories.

make_distribution(dist)

Generate a UnivariateDistribution class from a compatible object

Normal distribution with prescribed mean and standard deviation.

Uniform distribution.

Binomial(*, n, p, **kwargs)

Binomial distribution with prescribed success probability and number of trials

Mixture(components, *[, weights])

Representation of a mixture distribution.

order_statistic(X, /, *, r, n)

Probability distribution of an order statistic

truncate(X[, lb, ub])

Truncate the support of a random variable.

Absolute value of a random variable

Natural exponential of a random variable

Natural logarithm of a non-negative random variable

boxcox(x[, lmbda, alpha, optimizer])

Return a dataset transformed by a Box-Cox power transformation.

boxcox_normmax(x[, brack, method, ...])

Compute optimal Box-Cox transform parameter for input data.

boxcox_llf(lmb, data, *[, axis, keepdims, ...])

The boxcox log-likelihood function.

yeojohnson(x[, lmbda])

Return a dataset transformed by a Yeo-Johnson power transformation.

yeojohnson_normmax(x[, brack])

Compute optimal Yeo-Johnson transform parameter.

yeojohnson_llf(lmb, data)

The yeojohnson log-likelihood function.

obrientransform(*samples)

Compute the O'Brien transform on input data (any number of arrays).

sigmaclip(a[, low, high])

Perform iterative sigma-clipping of array elements.

trimboth(a, proportiontocut[, axis])

Slice off a proportion of items from both ends of an array.

trim1(a, proportiontocut[, tail, axis])

Slice off a proportion from ONE end of the passed array distribution.

zmap(scores, compare[, axis, ddof, nan_policy])

Calculate the relative z-scores.

zscore(a[, axis, ddof, nan_policy])

gzscore(a, *[, axis, ddof, nan_policy])

Compute the geometric standard score.

wasserstein_distance(u_values, v_values[, ...])

Compute the Wasserstein-1 distance between two 1D discrete distributions.

wasserstein_distance_nd(u_values, v_values)

Compute the Wasserstein-1 distance between two N-D discrete distributions.

energy_distance(u_values, v_values[, ...])

Compute the energy distance between two 1D distributions.

fit(dist, data[, bounds, guess, method, ...])

Fit a discrete or continuous distribution to data

Empirical cumulative distribution function of a sample.

logrank(x, y[, alternative])

Compare the survival distributions of two samples via the logrank test.

directional_stats(samples, *[, axis, normalize])

Computes sample statistics for directional data.

circmean(samples[, high, low, axis, ...])

Compute the circular mean of a sample of angle observations.

circvar(samples[, high, low, axis, ...])

Compute the circular variance of a sample of angle observations.

circstd(samples[, high, low, axis, ...])

Compute the circular standard deviation of a sample of angle observations.

sobol_indices(*, func, n[, dists, method, ...])

Global sensitivity indices of Sobol'.

ppcc_max(x[, brack, dist])

Calculate the shape parameter that maximizes the PPCC.

ppcc_plot(x, a, b[, dist, plot, N])

Calculate and optionally plot probability plot correlation coefficient.

probplot(x[, sparams, dist, fit, plot, rvalue])

Calculate quantiles for a probability plot, and optionally show the plot.

boxcox_normplot(x, la, lb[, plot, N])

Compute parameters for a Box-Cox normality plot, optionally show it.

yeojohnson_normplot(x, la, lb[, plot, N])

Compute parameters for a Yeo-Johnson normality plot, optionally show it.

gaussian_kde(dataset[, bw_method, weights])

Representation of a kernel-density estimate using Gaussian kernels.

DegenerateDataWarning([msg])

Warns when data is degenerate and results may not be reliable.

ConstantInputWarning([msg])

Warns when all values in data are exactly equal.

NearConstantInputWarning([msg])

Warns when all values in data are nearly equal.

Represents an error condition when fitting a distribution to data.

These classes are private, but they are included here because instances of them are returned by other statistical functions. User import and instantiation is not supported.

---

## rsf2csf#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.rsf2csf.html

**Contents:**
- rsf2csf#

Convert real Schur form to complex Schur form.

Convert a quasi-diagonal real-valued Schur form to the upper-triangular complex-valued Schur form.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Real Schur form of the original array

Schur transformation matrix

Whether to check that the input arrays contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Complex Schur form of the original array

Schur transformation matrix corresponding to the complex form

Schur decomposition of an array

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import schur, rsf2csf
>>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
>>> T, Z = schur(A)
>>> T
array([[ 2.65896708,  1.42440458, -1.92933439],
       [ 0.        , -0.32948354, -0.49063704],
       [ 0.        ,  1.31178921, -0.32948354]])
>>> Z
array([[0.72711591, -0.60156188, 0.33079564],
       [0.52839428, 0.79801892, 0.28976765],
       [0.43829436, 0.03590414, -0.89811411]])
>>> T2 , Z2 = rsf2csf(T, Z)
>>> T2
array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
       [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
       [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
>>> Z2
array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
       [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
       [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])
```

---

## qr#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html

**Contents:**
- qr#

Compute QR decomposition of a matrix.

Calculate the decomposition A = Q R where Q is unitary/orthogonal and R upper triangular.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Matrix to be decomposed

Whether data in a is overwritten (may improve performance if overwrite_a is set to True by reusing the existing input data structure rather than creating a new one.)

Work array size, lwork >= a.shape[1]. If None or -1, an optimal size is computed.

Determines what information is to be returned: either both Q and R (‘full’, default), only R (‘r’) or both Q and R but computed in economy-size (‘economic’, see Notes). The final option ‘raw’ (added in SciPy 0.11) makes the function return two matrices (Q, TAU) in the internal format used by LAPACK.

Whether or not factorization should include pivoting for rank-revealing qr decomposition. If pivoting, compute the decomposition A[:, P] = Q @ R as above, but where P is chosen such that the diagonal of R is non-increasing. Equivalently, albeit less efficiently, an explicit P matrix may be formed explicitly by permuting the rows or columns (depending on the side of the equation on which it is to be used) of an identity matrix. See Examples.

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Of shape (M, M), or (M, K) for mode='economic'. Not returned if mode='r'. Replaced by tuple (Q, TAU) if mode='raw'.

Of shape (M, N), or (K, N) for mode in ['economic', 'raw']. K = min(M, N).

Of shape (N,) for pivoting=True. Not returned if pivoting=False.

Raised if decomposition fails

This is an interface to the LAPACK routines dgeqrf, zgeqrf, dorgqr, zungqr, dgeqp3, and zgeqp3.

If mode=economic, the shapes of Q and R are (M, K) and (K, N) instead of (M,M) and (M,N), with K=min(M,N).

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> rng = np.random.default_rng()
>>> a = rng.standard_normal((9, 6))
```

Example 2 (unknown):
```unknown
>>> q, r = linalg.qr(a)
>>> np.allclose(a, np.dot(q, r))
True
>>> q.shape, r.shape
((9, 9), (9, 6))
```

Example 3 (unknown):
```unknown
>>> r2 = linalg.qr(a, mode='r')
>>> np.allclose(r, r2)
True
```

Example 4 (unknown):
```unknown
>>> q3, r3 = linalg.qr(a, mode='economic')
>>> q3.shape, r3.shape
((9, 6), (6, 6))
```

---

## SciPy User Guide#

**URL:** https://docs.scipy.org/doc/scipy/tutorial/index.html

**Contents:**
- SciPy User Guide#
- Subpackages and User Guides#

SciPy is a collection of mathematical algorithms and convenience functions built on NumPy . It adds significant power to Python by providing the user with high-level commands and classes for manipulating and visualizing data.

SciPy is organized into subpackages covering different scientific computing domains. These are summarized in the following table, with their user guide linked in the Description and User Guide column (if available):

Description and User Guide

Clustering algorithms

Physical and mathematical constants

Finite difference differentiation tools

Fourier Transforms (scipy.fft)

Fast Fourier Transform routines (legacy)

Integration (scipy.integrate)

Interpolation (scipy.interpolate)

Linear Algebra (scipy.linalg)

Multidimensional Image Processing (scipy.ndimage)

Orthogonal distance regression

Optimization (scipy.optimize)

Signal Processing (scipy.signal)

Sparse Arrays (scipy.sparse)

Spatial Data Structures and Algorithms (scipy.spatial)

Special Functions (scipy.special)

Statistics (scipy.stats)

There are also additional user guides for these topics:

Sparse eigenvalue problems with ARPACK - Eigenvalue problem solver using iterative methods

Compressed Sparse Graph Routines (scipy.sparse.csgraph) - Compressed Sparse Graph Routines

For guidance on organizing and importing functions from SciPy subpackages, refer to the Guidelines for Importing Functions from SciPy.

For information on support for parallel execution and thread safety, see Parallel execution support in SciPy and Thread Safety in SciPy.

---

## median_filter#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

**Contents:**
- median_filter#

Calculate a multidimensional median filter.

See footprint, below. Ignored if footprint is given.

Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function. footprint is a boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)). We adjust size to the number of dimensions of the input array, so that, if the input array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When footprint is given, size is ignored.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right. By passing a sequence of origins with length equal to the number of dimensions of the input array, different shifts can be specified along each axis.

If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes. When axes is specified, any tuples used for size, origin, and/or mode must match the length of axes. The ith entry in any of these tuples corresponds to the ith entry in axes.

Filtered array. Has the same shape as input.

For 2-dimensional images with uint8, float32 or float64 dtypes the specialised function scipy.signal.medfilt2d may be faster. It is however limited to constant mode with cval=0.

The filter always returns the argument that would appear at index n // 2 in a sorted array, where n is the number of elements in the footprint of the filter. Note that this differs from the conventional definition of the median when n is even. Also, this function does not support the float16 dtype, behavior in the presence of NaNs is undefined, and memory consumption scales with n**4. For float16 support, greater control over the definition of the filter, and to limit memory usage, consider using vectorized_filter with NumPy functions np.median or np.nanmedian.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> plt.gray()  # show the filtered result in grayscale
>>> ax1 = fig.add_subplot(121)  # left side
>>> ax2 = fig.add_subplot(122)  # right side
>>> ascent = datasets.ascent()
>>> result = ndimage.median_filter(ascent, size=20)
>>> ax1.imshow(ascent)
>>> ax2.imshow(result)
>>> plt.show()
```

---

## funm#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.funm.html

**Contents:**
- funm#

Evaluate a matrix function specified by a callable.

Returns the value of matrix-valued function f at A. The function f is an extension of the scalar-valued function func to matrices.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Matrix at which to evaluate the function

Callable object that evaluates a scalar function f. Must be vectorized (eg. using vectorize).

Print warning if error in the result is estimated large instead of returning estimated error. (Default: True)

Value of the matrix function specified by func evaluated at A

1-norm of the estimated error, ||err||_1 / ||A||_1

This function implements the general algorithm based on Schur decomposition (Algorithm 9.1.1. in [1]).

If the input matrix is known to be diagonalizable, then relying on the eigendecomposition is likely to be faster. For example, if your matrix is Hermitian, you can do

Gene H. Golub, Charles F. van Loan, Matrix Computations 4th ed.

**Examples:**

Example 1 (python):
```python
>>> from scipy.linalg import eigh
>>> def funm_herm(a, func, check_finite=False):
...     w, v = eigh(a, check_finite=check_finite)
...     ## if you further know that your matrix is positive semidefinite,
...     ## you can optionally guard against precision errors by doing
...     # w = np.maximum(w, 0)
...     w = func(w)
...     return (v * w).dot(v.conj().T)
```

Example 2 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import funm
>>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
>>> funm(a, lambda x: x*x)
array([[  4.,  15.],
       [  5.,  19.]])
>>> a.dot(a)
array([[  4.,  15.],
       [  5.,  19.]])
```

---

## svd#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

**Contents:**
- svd#

Singular Value Decomposition.

Factorizes the matrix a into two unitary matrices U and Vh, and a 1-D array s of singular values (real, non-negative) such that a == U @ S @ Vh, where S is a suitably shaped matrix of zeros with main diagonal s.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

If True (default), U and Vh are of shape (M, M), (N, N). If False, the shapes are (M, K) and (K, N), where K = min(M, N).

Whether to compute also U and Vh in addition to s. Default is True.

Whether to overwrite a; may improve performance. Default is False.

Whether to check that the input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Whether to use the more efficient divide-and-conquer approach ('gesdd') or general rectangular approach ('gesvd') to compute the SVD. MATLAB and Octave use the 'gesvd' approach. Default is 'gesdd'.

Unitary matrix having left singular vectors as columns. Of shape (M, M) or (M, K), depending on full_matrices.

The singular values, sorted in non-increasing order. Of shape (K,), with K = min(M, N).

Unitary matrix having right singular vectors as rows. Of shape (N, N) or (K, N) depending on full_matrices.

If SVD computation does not converge.

Compute singular values of a matrix.

Construct the Sigma matrix, given the vector s.

Reconstruct the original matrix from the decomposition:

Alternatively, use full_matrices=False (notice that the shape of U is then (m, n) instead of (m, m)):

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> from scipy import linalg
>>> rng = np.random.default_rng()
>>> m, n = 9, 6
>>> a = rng.standard_normal((m, n)) + 1.j*rng.standard_normal((m, n))
>>> U, s, Vh = linalg.svd(a)
>>> U.shape,  s.shape, Vh.shape
((9, 9), (6,), (6, 6))
```

Example 2 (bash):
```bash
>>> sigma = np.zeros((m, n))
>>> for i in range(min(m, n)):
...     sigma[i, i] = s[i]
>>> a1 = np.dot(U, np.dot(sigma, Vh))
>>> np.allclose(a, a1)
True
```

Example 3 (unknown):
```unknown
>>> U, s, Vh = linalg.svd(a, full_matrices=False)
>>> U.shape, s.shape, Vh.shape
((9, 6), (6,), (6, 6))
>>> S = np.diag(s)
>>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
True
```

Example 4 (unknown):
```unknown
>>> s2 = linalg.svd(a, compute_uv=False)
>>> np.allclose(s, s2)
True
```

---

## prewitt#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.prewitt.html

**Contents:**
- prewitt#

Calculate a Prewitt filter.

The axis of input along which to calculate. Default is -1.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended when the filter overlaps a border. By passing a sequence of modes with length equal to the number of dimensions of the input array, different modes can be specified along each axis. Default value is ‘reflect’. The valid values and their behavior is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘constant’.

This is a synonym for ‘reflect’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Filtered array. Has the same shape as input.

This function computes the one-dimensional Prewitt filter. Horizontal edges are emphasised with the horizontal transform (axis=0), vertical edges with the vertical transform (axis=1), and so on for higher dimensions. These can be combined to give the magnitude.

**Examples:**

Example 1 (python):
```python
>>> from scipy import ndimage, datasets
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> ascent = datasets.ascent()
>>> prewitt_h = ndimage.prewitt(ascent, axis=0)
>>> prewitt_v = ndimage.prewitt(ascent, axis=1)
>>> magnitude = np.sqrt(prewitt_h ** 2 + prewitt_v ** 2)
>>> magnitude *= 255 / np.max(magnitude) # Normalization
>>> fig, axes = plt.subplots(2, 2, figsize = (8, 8))
>>> plt.gray()
>>> axes[0, 0].imshow(ascent)
>>> axes[0, 1].imshow(prewitt_h)
>>> axes[1, 0].imshow(prewitt_v)
>>> axes[1, 1].imshow(magnitude)
>>> titles = ["original", "horizontal", "vertical", "magnitude"]
>>> for i, ax in enumerate(axes.ravel()):
...     ax.set_title(titles[i])
...     ax.axis("off")
>>> plt.show()
```

---

## RectBivariateSpline#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html

**Contents:**
- RectBivariateSpline#

Bivariate spline approximation over a rectangular mesh.

Can be used for both smoothing and interpolating data.

1-D arrays of coordinates in strictly ascending order. Evaluated points outside the data range will be extrapolated.

2-D array of data with shape (x.size,y.size).

Sequence of length 4 specifying the boundary of the rectangular approximation domain, which means the start and end spline knots of each dimension are set by these values. By default, bbox=[min(x), max(x), min(y), max(y)].

Degrees of the bivariate spline. Default is 3.

Positive smoothing factor defined for estimation condition: sum((z[i]-f(x[i], y[i]))**2, axis=0) <= s where f is a spline function. Default is s=0, which is for interpolation.

The maximal number of iterations maxit allowed for finding a smoothing spline with fp=s. Default is maxit=20.

__call__(x, y[, dx, dy, grid])

Evaluate the spline or its derivatives at given positions.

Evaluate the spline at points

Return spline coefficients.

Return a tuple (tx,ty) where tx,ty contain knots positions of the spline with respect to x-, y-variable, respectively.

Return weighted sum of squared residuals of the spline approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)

integral(xa, xb, ya, yb)

Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

partial_derivative(dx, dy)

Construct a new spline representing a partial derivative of this spline.

a base class for bivariate splines.

a smooth univariate spline to fit a given set of data points.

a smoothing bivariate spline through the given points

a bivariate spline using weighted least-squares fitting

a bivariate spline over a rectangular mesh on a sphere

a smoothing bivariate spline in spherical coordinates

a bivariate spline in spherical coordinates using weighted least-squares fitting

a function to find a bivariate B-spline representation of a surface

a function to evaluate a bivariate B-spline and its derivatives

If the input data is such that input dimensions have incommensurate units and differ by many orders of magnitude, the interpolant may have numerical artifacts. Consider rescaling the data before interpolating.

---

## Radau#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.Radau.html

**Contents:**
- Radau#

Implicit Runge-Kutta method of Radau IIA family of order 5.

The implementation follows [1]. The error is controlled with a third-order accurate embedded formula. A cubic polynomial which satisfies the collocation conditions is used for the dense output.

Right-hand side of the system: the time derivative of the state y at time t. The calling signature is fun(t, y), where t is a scalar and y is an ndarray with len(y) = len(y0). fun must return an array of the same shape as y. See vectorized for more information.

Boundary time - the integration won’t continue beyond it. It also determines the direction of the integration.

Initial step size. Default is None which means that the algorithm should choose.

Maximum allowed step size. Default is np.inf, i.e., the step size is not bounded and determined solely by the solver.

Relative and absolute tolerances. The solver keeps the local error estimates less than atol + rtol * abs(y). HHere rtol controls a relative accuracy (number of correct digits), while atol controls absolute accuracy (number of correct decimal places). To achieve the desired rtol, set atol to be smaller than the smallest value that can be expected from rtol * abs(y) so that rtol dominates the allowable error. If atol is larger than rtol * abs(y) the number of correct digits is not guaranteed. Conversely, to achieve the desired atol set rtol such that rtol * abs(y) is always smaller than atol. If components of y have different scales, it might be beneficial to set different atol values for different components by passing array_like with shape (n,) for atol. Default values are 1e-3 for rtol and 1e-6 for atol.

Jacobian matrix of the right-hand side of the system with respect to y, required by this method. The Jacobian matrix has shape (n, n) and its element (i, j) is equal to d f_i / d y_j. There are three ways to define the Jacobian:

If array_like or sparse_matrix, the Jacobian is assumed to be constant.

If callable, the Jacobian is assumed to depend on both t and y; it will be called as jac(t, y) as necessary. For the ‘Radau’ and ‘BDF’ methods, the return value might be a sparse matrix.

If None (default), the Jacobian will be approximated by finite differences.

It is generally recommended to provide the Jacobian rather than relying on a finite-difference approximation.

Defines a sparsity structure of the Jacobian matrix for a finite-difference approximation. Its shape must be (n, n). This argument is ignored if jac is not None. If the Jacobian has only few non-zero elements in each row, providing the sparsity structure will greatly speed up the computations [2]. A zero entry means that a corresponding element in the Jacobian is always zero. If None (default), the Jacobian is assumed to be dense.

Whether fun can be called in a vectorized fashion. Default is False.

If vectorized is False, fun will always be called with y of shape (n,), where n = len(y0).

If vectorized is True, fun may be called with y of shape (n, k), where k is an integer. In this case, fun must behave such that fun(t, y)[:, i] == fun(t, y[:, i]) (i.e. each column of the returned array is the time derivative of the state corresponding with a column of y).

Setting vectorized=True allows for faster finite difference approximation of the Jacobian by this method, but may result in slower execution overall in some circumstances (e.g. small len(y0)).

Current status of the solver: ‘running’, ‘finished’ or ‘failed’.

Integration direction: +1 or -1.

Previous time. None if no steps were made yet.

Size of the last successful step. None if no steps were made yet.

Number of evaluations of the right-hand side.

Number of evaluations of the Jacobian.

Number of LU decompositions.

Compute a local interpolant over the last successful step.

Perform one integration step.

E. Hairer, G. Wanner, “Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems”, Sec. IV.8.

A. Curtis, M. J. D. Powell, and J. Reid, “On the estimation of sparse Jacobian matrices”, Journal of the Institute of Mathematics and its Applications, 13, pp. 117-120, 1974.

---

## cholesky#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky.html

**Contents:**
- cholesky#

Compute the Cholesky decomposition of a matrix.

Returns the Cholesky decomposition, \(A = L L^*\) or \(A = U^* U\) of a Hermitian positive-definite matrix A.

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Matrix to be decomposed

Whether to compute the upper- or lower-triangular Cholesky factorization. During decomposition, only the selected half of the matrix is referenced. Default is upper-triangular.

Whether to overwrite data in a (may improve performance).

Whether to check that the entire input matrix contains only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Upper- or lower-triangular Cholesky factor of a.

During the finiteness check (if selected), the entire matrix a is checked. During decomposition, a is assumed to be symmetric or Hermitian (as applicable), and only the half selected by option lower is referenced. Consequently, if a is asymmetric/non-Hermitian, cholesky may still succeed if the symmetric/Hermitian matrix represented by the selected half is positive definite, yet it may fail if an element in the other half is non-finite.

**Examples:**

Example 1 (sql):
```sql
>>> import numpy as np
>>> from scipy.linalg import cholesky
>>> a = np.array([[1,-2j],[2j,5]])
>>> L = cholesky(a, lower=True)
>>> L
array([[ 1.+0.j,  0.+0.j],
       [ 0.+2.j,  1.+0.j]])
>>> L @ L.T.conj()
array([[ 1.+0.j,  0.-2.j],
       [ 0.+2.j,  5.+0.j]])
```

---

## RK45#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html

**Contents:**
- RK45#

Explicit Runge-Kutta method of order 5(4).

This uses the Dormand-Prince pair of formulas [1]. The error is controlled assuming accuracy of the fourth-order method accuracy, but steps are taken using the fifth-order accurate formula (local extrapolation is done). A quartic interpolation polynomial is used for the dense output [2].

Can be applied in the complex domain.

Right-hand side of the system. The calling signature is fun(t, y). Here t is a scalar, and there are two options for the ndarray y: It can either have shape (n,); then fun must return array_like with shape (n,). Alternatively it can have shape (n, k); then fun must return an array_like with shape (n, k), i.e., each column corresponds to a single column in y. The choice between the two options is determined by vectorized argument (see below).

Boundary time - the integration won’t continue beyond it. It also determines the direction of the integration.

Initial step size. Default is None which means that the algorithm should choose.

Maximum allowed step size. Default is np.inf, i.e., the step size is not bounded and determined solely by the solver.

Relative and absolute tolerances. The solver keeps the local error estimates less than atol + rtol * abs(y). Here rtol controls a relative accuracy (number of correct digits), while atol controls absolute accuracy (number of correct decimal places). To achieve the desired rtol, set atol to be smaller than the smallest value that can be expected from rtol * abs(y) so that rtol dominates the allowable error. If atol is larger than rtol * abs(y) the number of correct digits is not guaranteed. Conversely, to achieve the desired atol set rtol such that rtol * abs(y) is always smaller than atol. If components of y have different scales, it might be beneficial to set different atol values for different components by passing array_like with shape (n,) for atol. Default values are 1e-3 for rtol and 1e-6 for atol.

Whether fun is implemented in a vectorized fashion. Default is False.

Current status of the solver: ‘running’, ‘finished’ or ‘failed’.

Integration direction: +1 or -1.

Previous time. None if no steps were made yet.

Size of the last successful step. None if no steps were made yet.

Number evaluations of the system’s right-hand side.

Number of evaluations of the Jacobian. Is always 0 for this solver as it does not use the Jacobian.

Number of LU decompositions. Is always 0 for this solver.

Compute a local interpolant over the last successful step.

Perform one integration step.

J. R. Dormand, P. J. Prince, “A family of embedded Runge-Kutta formulae”, Journal of Computational and Applied Mathematics, Vol. 6, No. 1, pp. 19-26, 1980.

L. W. Shampine, “Some Practical Runge-Kutta Formulas”, Mathematics of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.

---

## solve#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html

**Contents:**
- solve#

Solve the equation a @ x = b for x, where a is a square matrix.

If the data matrix is known to be a particular type then supplying the corresponding string to assume_a key chooses the dedicated solver. The available options are

‘symmetric’ (or ‘sym’)

‘hermitian’ (or ‘her’)

symmetric positive definite

‘positive definite’ (or ‘pos’)

The documentation is written assuming array arguments are of specified “core” shapes. However, array argument(s) of this function may have additional “batch” dimensions prepended to the core shape. In this case, the array is treated as a batch of lower-dimensional slices; see Batched Linear Operations for details.

Input data for the right hand side.

Ignored unless assume_a is one of 'sym', 'her', or 'pos'. If True, the calculation uses only the data in the lower triangle of a; entries above the diagonal are ignored. If False (default), the calculation uses only the data in the upper triangle of a; entries below the diagonal are ignored.

Allow overwriting data in a (may enhance performance).

Allow overwriting data in b (may enhance performance).

Whether to check that the input matrices contain only finite numbers. Disabling may give a performance gain, but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.

Valid entries are described above. If omitted or None, checks are performed to identify structure so the appropriate solver can be called.

If True, solve a.T @ x == b. Raises NotImplementedError for complex a.

If size mismatches detected or input a is not square.

If the computation fails because of matrix singularity.

If an ill-conditioned input a is detected.

If transposed is True and input a is a complex matrix.

If the input b matrix is a 1-D array with N elements, when supplied together with an NxN input a, it is assumed as a valid column vector despite the apparent size mismatch. This is compatible with the numpy.dot() behavior and the returned result is still 1-D array.

The general, symmetric, Hermitian and positive definite solutions are obtained via calling ?GESV, ?SYSV, ?HESV, and ?POSV routines of LAPACK respectively.

The datatype of the arrays define which solver is called regardless of the values. In other words, even when the complex array entries have precisely zero imaginary parts, the complex solver will be called based on the data type of the array.

Given a and b, solve for x:

**Examples:**

Example 1 (python):
```python
>>> import numpy as np
>>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
>>> b = np.array([2, 4, -1])
>>> from scipy import linalg
>>> x = linalg.solve(a, b)
>>> x
array([ 2., -2.,  9.])
>>> np.dot(a, x) == b
array([ True,  True,  True], dtype=bool)
```

---

## maximum_filter1d#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter1d.html

**Contents:**
- maximum_filter1d#

Calculate a 1-D maximum filter along the given axis.

The lines of the array along the given axis are filtered with a maximum filter of given size.

Length along which to calculate the 1-D maximum.

The axis of input along which to calculate. Default is -1.

The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.

The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:

The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

The input is extended by replicating the last pixel.

The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

The input is extended by wrapping around to the opposite edge.

For consistency with the interpolation functions, the following mode names can also be used:

This is a synonym for ‘reflect’.

This is a synonym for ‘constant’.

This is a synonym for ‘wrap’.

Value to fill past edges of input if mode is ‘constant’. Default is 0.0.

Controls the placement of the filter on the input array’s pixels. A value of 0 (the default) centers the filter over the pixel, with positive values shifting the filter to the left, and negative ones to the right.

Maximum-filtered array with same shape as input. None if output is not None

This function implements the MAXLIST algorithm [1], as described by Richard Harter [2], and has a guaranteed O(n) performance, n being the input length, regardless of filter size.

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777

http://www.richardhartersworld.com/cri/2001/slidingmin.html

**Examples:**

Example 1 (sql):
```sql
>>> from scipy.ndimage import maximum_filter1d
>>> maximum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
array([8, 8, 8, 4, 9, 9, 9, 9])
```

---

## ifft#

**URL:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifft.html

**Contents:**
- ifft#

Compute the 1-D inverse discrete Fourier Transform.

This function computes the inverse of the 1-D n-point discrete Fourier transform computed by fft. In other words, ifft(fft(x)) == x to within numerical accuracy.

The input should be ordered in the same way as is returned by fft, i.e.,

x[0] should contain the zero frequency term,

x[1:n//2] should contain the positive-frequency terms,

x[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.

For an even number of input points, x[n//2] represents the sum of the values at the positive and negative Nyquist frequencies, as the two are aliased together. See fft for details.

Input array, can be complex.

Length of the transformed axis of the output. If n is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros. If n is not given, the length of the input along the axis specified by axis is used. See notes about padding issues.

Axis over which to compute the inverse DFT. If not given, the last axis is used.

Normalization mode (see fft). Default is “backward”.

If True, the contents of x can be destroyed; the default is False. See fft for more details.

Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). See fft for more details.

This argument is reserved for passing in a precomputed plan provided by downstream FFT vendors. It is currently not used in SciPy.

Added in version 1.5.0.

The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified.

If axes is larger than the last axis of x.

The 1-D (forward) FFT, of which ifft is the inverse.

If the input parameter n is larger than the size of the input, the input is padded by appending zeros at the end. Even though this is the common approach, it might lead to surprising results. If a different padding is desired, it must be performed before calling ifft.

If x is a 1-D array, then the ifft is equivalent to

As with fft, ifft has support for all floating point types and is optimized for real input.

Create and plot a band-limited signal with random phases:

**Examples:**

Example 1 (unknown):
```unknown
y[k] = np.sum(x * np.exp(2j * np.pi * k * np.arange(n)/n)) / len(x)
```

Example 2 (typescript):
```typescript
>>> import scipy.fft
>>> import numpy as np
>>> scipy.fft.ifft([0, 4, 0, 0])
array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j]) # may vary
```

Example 3 (json):
```json
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> t = np.arange(400)
>>> n = np.zeros((400,), dtype=complex)
>>> n[40:60] = np.exp(1j*rng.uniform(0, 2*np.pi, (20,)))
>>> s = scipy.fft.ifft(n)
>>> plt.plot(t, s.real, 'b-', t, s.imag, 'r--')
[<matplotlib.lines.Line2D object at ...>, <matplotlib.lines.Line2D object at ...>]
>>> plt.legend(('real', 'imaginary'))
<matplotlib.legend.Legend object at ...>
>>> plt.show()
```

---
