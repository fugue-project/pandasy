class SlideException(Exception):
    """General Slide level exception"""

    pass


class SlideInvalidOperation(SlideException):
    """Invalid operations"""

    pass


class SlideCastError(SlideException):
    """Type casting exception"""

    pass
