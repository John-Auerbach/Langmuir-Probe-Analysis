In a single Langmuir probe measurement, the probe is biased relative to the surrounding plasma and the total collected current is the difference between ion and electron collection. The measured current I(V) is written as

$$
I(V) = I_{\rm i}(V) - I_{\rm e}(V),
$$

where $I_{\rm i}$ is the ion current collected by the probe and $I_{\rm e}$ is the electron current. For a sufficiently negative probe bias, the ion current is virtually independent of small changes in probe potential and can be treated as a saturation current, $I_{\rm i}(V) \approx I_{\rm is}$. Hence in the ion-saturation region, we have

$$
I(V) \approx I_{\rm is} \quad (V\;\text{negative}).
$$

The bias dependence of the electron current in the retarding region is approximately exponential, given by

$$
I_{\rm e}(V) \approx I_{\rm es}\,\mathrm{exp}\left[\frac{-e V}{k_{\rm B}T_{\rm e}}\right],
$$

where $I_{\rm es}$ is the electron saturation current, $e$ is the elementary charge, $k_{\rm B}$ Boltzmann's constant and $T_{\rm e}$ the electron temperature. Combining the above and using the ion-saturation approximation in the negative-bias region gives

$$
I(V) = I_{\rm is} - I_{\rm es}\,\mathrm{exp}\left[\frac{-e V}{k_{\rm B}T_{\rm e}}\right].
$$

In the code, I determine $I_{\rm is}$ by performing a linear regression on the measured I–V points where the probe bias is negative (ion-dominated region). I then extend that fitted ion contribution across the entire bias range and subtract it from the total measured current to obtain the isolated electron current

$$
I_{\rm e}(V) = I(V) - I_{\rm ion\_fit}(V) .
$$

The floating potential $V_{\rm f}$ is defined where the net probe current vanishes, $I(V_{\rm f})=0$. I added interpolation to help find the exact $V_{\rm f}$. The plasma potential $V_{\rm p}$ is identified from the maximum in electron collection and is estimated numerically as the bias where $-\mathrm{d}I/\mathrm{d}V$ is minimal.

In the electron-retarding region (between $V_{\rm f}$ and $V_{\rm p}$) the isolated electron current follows the exponential model above. Taking the natural logarithm yields a near-linear relation

$$
\ln I_{\rm e}(V) \approx \frac{ -e}{k_{\rm B}T_{\rm e}}\,V + \ln I_{\rm es},
$$

where electron temperature can be deduced from the slope:

$$
T_{\rm e}=\frac{-e}{\rm slope \cdot \textit k_{\rm B}}.
$$

From the fit intercept we recover the electron saturation current $I_{\rm es}=\exp(\rm intercept)$. The electron density is then obtained from the electron saturation current equation

$$
I_{\rm es} = e A n_{\rm e} \sqrt{\frac{k_{\rm B}T_{\rm e,J}}{2\pi m_{\rm e}}},
$$

where $A$ is the probe area, $m_{\rm e}$ the electron mass, and $k_{\rm B}T_{\rm e,J}$ is the electron energy expressed in joules (i.e. $k_{\rm B}T_{\rm e}$ in eV multiplied by $1.60218\times10^{-19}$ J/eV). Rearranging the expression yields

$$
n_{\rm e} = \frac{I_{\rm es}}{e A}\sqrt{\frac{2\pi m_{\rm e}}{k_{\rm B}T_{\rm e,J}}} .
$$

Finally, the Debye length is calculated from

$$
\lambda_{\rm D} = \sqrt{\frac{\varepsilon_0 k_{\rm B}T_{\rm e,J}}{n_{\rm e} e^2}}\;.
$$