


from aenum import Enum



class DebugLevel(Enum):
    VERBOSE = 0x00000001,   'Verbose'
    LIGHT   = 0x00000002,   'Light'
    RELEASE = 0x00000003,   'Release'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value



class Mode(Enum):
    TRAIN = 0x00000011,
    TEST = 0x00000012