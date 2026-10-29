"""Check group weighting and sample-weight routing for independent targets."""
import unittest
import numpy as np
from refine_expanded_rf import group_weights,candidate_models,fit_model

class RefinementTests(unittest.TestCase):
    def test_equal_group_total_weight(self):
        g=np.array(['a','a','a','b','c','c'])
        w=group_weights(g)
        self.assertAlmostEqual(w.mean(),1)
        totals=[w[g==name].sum() for name in ['a','b','c']]
        np.testing.assert_allclose(totals,totals[0])
        self.assertTrue((w>0).all())

    def test_weighting_routes_through_both_target_models(self):
        import pandas as pd
        X=pd.DataFrame({'ppg_demo':np.arange(30,dtype=float)})
        y=np.column_stack([np.arange(30)+100,np.arange(30)/2+60])
        g=np.repeat(np.arange(10),3)
        template,balanced=candidate_models()['Group-balanced separate RF']
        template.set_params(regressor__forest__estimator__n_estimators=2)
        model=fit_model(template,X,y,g,balanced)
        self.assertEqual(model.predict(X).shape,y.shape)
        self.assertTrue(np.isfinite(model.predict(X)).all())

if __name__=='__main__':unittest.main()
