from glmpowercalc.exceptions.glmpowercalc_exception import GlmPowerCalcException

class RanksymmValidationException(GlmPowerCalcException):
    """Input Validation Exceptions for the ranksymm function"""

    def __init__(self, message):
        self.message = message