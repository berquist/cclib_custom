# -*- coding: utf-8 -*-

"""data_keepall.py: A ccData object that doesn't reject or filter for
invalid or empty attributes.

This allows for writing parsers that can parse/store/return arbitrary
things from output files just by extending cclib's builtin parser
classes and still parsing the 'canonical' attributes.
"""

from cclib.parser import ccData


class ccDataKeepall(ccData):

    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the superclass
        super(ccDataKeepall, self).__init__(*args, **kwargs)

    def setattributes(self, attributes):
        """Sets data attributes given in a dictionary.

        Inputs:
            attributes - dictionary of attributes to set
        Outputs:
            invalid - list of attributes names that were not set, which
                      means they are not specified in self._attrlist

        We return the empty list, because we no longer filter for
        invalid or empty attributes.
        """

        if type(attributes) is not dict:
            raise TypeError("attributes must be in a dictionary")

        ## We keep all attributes, with no distinction between
        ## valid/invalid/empty.

        # valid = [a for a in attributes if a in self._attrlist]
        # invalid = [a for a in attributes if a not in self._attrlist]

        # for attr in valid:
        for attr in attributes:
            setattr(self, attr, attributes[attr])

        self.arrayify()
        self.typecheck()

        return list()
