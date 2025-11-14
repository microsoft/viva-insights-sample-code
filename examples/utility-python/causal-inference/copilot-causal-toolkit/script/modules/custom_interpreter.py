import numpy as np
import logging
from econml.cate_interpreter import SingleTreeCateInterpreter
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

class CustomSingleTreeCateInterpreter(SingleTreeCateInterpreter):
    def interpret(self, cate_estimator, X, T0=None, T1=None, person_ids=None):
        """
        Interpret the heterogeneity of a CATE estimator for a specific T0/T1 comparison.

        Parameters
        ----------
        cate_estimator : EconML estimator (e.g., LinearDML)
            The fitted estimator to interpret.

        X : array_like
            The features for which to interpret the treatment effect.
            
        T0 : float, optional
            The baseline treatment level. If None, uses 5th percentile of treatment data.
            
        T1 : float, optional
            The comparison treatment level. If None, uses 95th percentile of treatment data.
            
        person_ids : array_like, optional
            PersonID array matching X rows for unique person counting.

        Returns
        -------
        self : object
            Fitted interpreter with .tree_model_ and .node_dict_ attributes.
        """
        # If T0 or T1 not provided, we need access to the original treatment data
        # For now, use default values that can be overridden
        if T0 is None:
            T0 = 1  # Will be updated to use percentiles when we have access to treatment data
        if T1 is None:
            T1 = 6  # Will be updated to use percentiles when we have access to treatment data
            
        # Store the T0/T1 values for reference
        self.T0_ = T0
        self.T1_ = T1
        
        # Store PersonID data for unique counting
        self.person_ids_ = person_ids
        
        self.tree_model_ = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease
        )

        y_pred = cate_estimator.effect(X, T0=T0, T1=T1)
        self.tree_model_.fit(X, y_pred.reshape((y_pred.shape[0], -1)))

        paths = self.tree_model_.decision_path(X)
        node_dict = {}
        for node_id in range(paths.shape[1]):
            mask = paths.getcol(node_id).toarray().flatten().astype(bool)
            Xsub = X[mask]

            if self.include_uncertainty and (
                not self.uncertainty_only_on_leaves or self.tree_model_.tree_.children_left[node_id] < 0
            ):
                res = cate_estimator.ate_inference(Xsub, T0=T0, T1=T1)
                
                # Calculate unique users for this node
                unique_users = self._count_unique_users_in_node(mask)
                
                # Extract p-value from inference result
                try:
                    p_value = float(res.pvalue())
                except Exception as e:
                    logger.warning(f"Could not extract p-value: {e}")
                    p_value = np.nan
                
                node_dict[node_id] = {
                    'mean': res.mean_point,
                    'std': res.std_point,
                    'ci': res.conf_int_mean(alpha=self.uncertainty_level),
                    'p_value': p_value,
                    'n_samples': len(Xsub),
                    'n_unique_users': unique_users
                }
            else:
                cate_node = y_pred[mask]
                
                # Calculate unique users for this node
                unique_users = self._count_unique_users_in_node(mask)
                
                node_dict[node_id] = {
                    'mean': np.mean(cate_node, axis=0),
                    'std': np.std(cate_node, axis=0),
                    'n_samples': len(Xsub),
                    'n_unique_users': unique_users
                }

        self.node_dict_ = node_dict
        return self
    
    def _count_unique_users_in_node(self, mask):
        """
        Count unique users in a tree node.
        
        Parameters
        ----------
        mask : array_like
            Boolean mask indicating which observations belong to this node
            
        Returns
        -------
        int
            Number of unique users in this node
        """
        if self.person_ids_ is not None:
            # Get PersonIDs for observations in this node
            person_ids_in_node = self.person_ids_[mask]
            # Count unique PersonIDs
            unique_users = len(np.unique(person_ids_in_node))
            return unique_users
        else:
            # Fallback: return observation count if no PersonID data
            return int(np.sum(mask))
    
    def get_tree_subgroups(self, feature_names):
        """
        Extract subgroups from the decision tree with their characteristics.
        
        Parameters
        ----------
        feature_names : list
            Names of the features used in the tree.
            
        Returns
        -------
        subgroups : list
            List of dictionaries containing subgroup information.
        """
        if not hasattr(self, 'tree_model_') or not hasattr(self, 'node_dict_'):
            raise ValueError("Interpreter must be fitted before extracting subgroups")
            
        tree = self.tree_model_.tree_
        subgroups = []
        
        def traverse_tree(node_id, condition_path=""):
            """Recursively traverse the tree to build condition paths."""
            node_info = self.node_dict_.get(node_id, {})
            
            # Check if this is a leaf node
            is_leaf = tree.children_left[node_id] == tree.children_right[node_id]
            
            if is_leaf:
                # This is a leaf node - create a subgroup
                group_id = f"Subgroup_{node_id}"
                
                # Use unique user count if available, otherwise fall back to observation count
                user_count = node_info.get('n_unique_users', node_info.get('n_samples', 0))
                
                subgroup = {
                    'group_id': group_id,
                    'filter_condition': condition_path.strip() if condition_path else "All Users",
                    'user_count': user_count,
                    'treatment_effect': float(node_info.get('mean', 0)),
                    'treatment_effect_std': float(node_info.get('std', 0)),
                    'node_id': node_id
                }
                
                # Add confidence interval if available
                if 'ci' in node_info:
                    ci = node_info['ci']
                    subgroup['ci_lower'] = float(ci[0]) if len(ci) > 0 else 0.0
                    subgroup['ci_upper'] = float(ci[1]) if len(ci) > 1 else 0.0
                
                # Add p-value if available
                if 'p_value' in node_info:
                    subgroup['p_value'] = float(node_info['p_value'])
                
                subgroups.append(subgroup)
            else:
                # Internal node - continue traversing
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
                
                # Left child (feature <= threshold)
                left_condition = f"{condition_path} & {feature_name} <= {threshold:.3f}" if condition_path else f"{feature_name} <= {threshold:.3f}"
                traverse_tree(tree.children_left[node_id], left_condition)
                
                # Right child (feature > threshold)  
                right_condition = f"{condition_path} & {feature_name} > {threshold:.3f}" if condition_path else f"{feature_name} > {threshold:.3f}"
                traverse_tree(tree.children_right[node_id], right_condition)
        
        # Start traversal from root
        traverse_tree(0)
        return subgroups