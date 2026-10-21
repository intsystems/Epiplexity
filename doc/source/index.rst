episynth
========

The practical goal is to train a time-series foundation model on high-quality
synthetic data and select productive data generators. A generator is productive
when its pretraining data improve forecasting on real datasets under a fixed
training-data budget. ``episynth`` compares generators through controlled training
and GIFT-Eval evaluation, and tests epiplexity-based measurements as selection
signals. The current experiments use a small Moirai model. The project repository is
`intsystems/Epiplexity <https://github.com/intsystems/Epiplexity>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   info
   installation
   demo

.. toctree::
   :maxdepth: 2
   :caption: Design and experiments

   architecture
   measurements
   stack
   extending
   experiments

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
