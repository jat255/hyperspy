import nose.tools
import numpy as np

from hyperspy.misc.eds import utils as utils_eds
from hyperspy.models.edstemmodel import EDSTEMModel as Model
from hyperspy.misc.elements import elements as elements_db


class TestlineFit:

    def setUp(self):
        s = utils_eds.xray_lines_model(elements=['Fe', 'Cr', 'Zn'],
                                       beam_energy=200,
                                       weight_percents=[20, 50, 30],
                                       energy_resolution_MnKa=130,
                                       energy_axis={'units': 'keV',
                                                    'size': 400,
                                                    'scale': 0.01,
                                                    'name': 'E',
                                                    'offset': 5.})
        s = s+0.002
        self.s = s

    def test_fit(self):
        s = self.s
        m = Model(s)
        m.fit()
        nose.tools.assert_true(np.allclose([i.data for i in
                                            m.get_lines_intensity()],
                                           [0.5, 0.2, 0.3], atol=1e-6))

    def test_calibrate_energy_axis(self):
        s = self.s
        m = Model(s)
        m.fit()
        reso = s.metadata.Acquisition_instrument.TEM.Detector.EDS.\
            energy_resolution_MnKa,
        s.set_microscope_parameters(energy_resolution_MnKa=150)
        m.calibrate_energy_axis(calibrate='resolution')
        nose.tools.assert_true(np.allclose(
            s.metadata.Acquisition_instrument.TEM.Detector.EDS.
            energy_resolution_MnKa, reso, atol=1))

    def test_calibrate_xray_energy(self):
        s = self.s
        m = Model(s)
        m.fit()
        m['Fe_Ka'].centre.value = 6.39
        m.calibrate_xray_lines(calibrate='energy', xray_lines=['Fe_Ka'],
                               bound=100)
        nose.tools.assert_true(np.allclose(
            m['Fe_Ka'].centre.value, elements_db['Fe']['Atomic_properties'][
                'Xray_lines']['Ka']['energy (keV)'], atol=1e-6))

    def test_calibrate_xray_weight(self):
        s = self.s
        s1 = utils_eds.xray_lines_model(
            elements=['Co'], energy_axis={'units': 'keV', 'size': 400,
                                          'scale': 0.01, 'name': 'E',
                                          'offset': 4.9})
        s = (s+s1/50)
        m = Model(s)
        m.fit()
        m.calibrate_xray_lines(calibrate='sub_weight',
                               xray_lines=['Fe_Ka'], bound=100)
        nose.tools.assert_true(np.allclose(0.0347, m['Fe_Kb'].A.value,
                               atol=1e-3))
