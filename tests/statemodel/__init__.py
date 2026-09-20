"""Reference state model for a finite slice of the traitlets contract.

This package intentionally does *not* import or reimplement ``HasTraits``.
It is a small, independently written table-driven model of the publicly
observable behaviour of descriptors, dynamic defaults, validators, observers,
notification holding and inheritance for the ``Int``, ``Unicode``,
``List`` and ``Union`` trait types.

The companion generator (:mod:`tests.statemodel.generator`) builds matching
class hierarchies on top of *both* this model and real traitlets, replays the
same operation sequences, and asserts that the observable traces are equal.
"""
