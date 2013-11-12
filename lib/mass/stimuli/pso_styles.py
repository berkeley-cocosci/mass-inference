class PSOStyler(object):

    def floor(self, pso, **kwargs):
        pso.setPos((0, 0, -0.5))

    def original(self, pso, **kwargs):
        pass

    def _mass(self, pso, **kwargs):
        blocktype = kwargs['blocktype']
        kappa = kwargs['kappa']

        if blocktype == 1:
            mass = pso.get_mass()
            mass *= (10 ** kappa)
            pso.set_mass(mass)

    def mass_red_yellow(self, pso, **kwargs):
        self._mass(pso, **kwargs)

    def mass_colors(self, pso, **kwargs):
        self._mass(pso, **kwargs)

    def mass_plastic_stone(self, pso, **kwargs):
        self._mass(pso, **kwargs)

    def apply(self, pso, style, **kwargs):
        style_func = getattr(self, style)
        style_func(pso, **kwargs)
