from modeling.evaluate import PlotResults

trial4_fine_results = PlotResults(test_type='fine', trial=4)
trial4_coarse_results = PlotResults(test_type='coarse', trial=4)

trial4_fine_results.plot_training_history()
trial4_fine_results.plot_confusion_matrix()
trial4_fine_results.plot_class_metrics()
