"""This module implements the Gaussian Mixture PHD filter for linear
   multi target tracking.."""

# Standard library
from copy import deepcopy

# Typing
from typing import List

# Mathematics
from numpy.linalg import inv
from numpy import ndarray

# Filter basis
#from pyrate.common.math import Gaussian
from pyrate_common_math_gaussian import Gaussian

class GaussianMixturePHD:

    """This class provides the gaussian mixture PHD filter (GMPHD) for linear multi target tracking."""

    # In this context, we reproduce a common filter notation
    # pylint: disable=invalid-name
    # pylint: disable=too-many-instance-attributes, too-many-arguments

    def __init__(
        self,
        birth_belief: List[Gaussian],
        survival_rate: float,
        detection_rate: float,
        intensity: float,
        F: ndarray,
        H: ndarray,
        Q: ndarray,
        R: ndarray,
    ):
        """The gaussian mixture PHD filter for linear multi-target tracking.

        The gaussian mixture PHD filter is a multi target tracker for linear state space models.
        It can be regarded as an extension of the Kalman filter formulas to so-called random
        finite sets (RFS). The PHD filter follows the same prediction-correction scheme for state
        estimation as the single target Kalman filters. As an additional part of the interface,
        the internal model for the filter's belief needs to be pruned regularly as to limit
        the computational complexity. The extraction of a state estimate is similarly more
        sophisticated in the PHD filter and requires the use of a dedicated procedure.

        Examples:
            Start by importing the necessary numpy functions.

            >>> from numpy import array
            >>> from numpy import eye
            >>> from numpy import vstack

            Setup the model.
            In this case, we track 1D positions with constant velocities.
            Thereby we choose the transition model like so.

            >>> F = array([[1.0, 1.0], [0.0, 0.0]])

            The measurements will be positions and no velocities.

            >>> H = array([[1.0, 0.0]])

            Furthermore, we assume the following noise on the process and measurements.

            >>> Q = eye(2)
            >>> R = eye(1)

            Our belief of how targets are generetaded is for them to start with
            a position and velocity of 0.

            >>> mean = vstack([0.0, 0.0])
            >>> covariance = array([[1.0, 0.0], [0.0, 1.0]])
            >>> birth_belief = [Gaussian(mean, covariance)]

            We need to tell the filter how certain we are to detect targets and whether they survive.
            Also, the amount of clutter in the observed environment is quantized.

            >>> survival_rate = 0.99
            >>> detection_rate = 0.99
            >>> intensity = 0.01

            Then, we initialize the filter. This model has not input, so we ignore B.

            >>> phd = GaussianMixturePHD(
            ...     birth_belief,
            ...     survival_rate,
            ...     detection_rate,
            ...     intensity,
            ...     F,
            ...     H,
            ...     Q,
            ...     R
            ... )

            We first predict with the provided model and then correct the prediction with a
            measurement of a single targets' position.

            >>> phd.predict()
            >>> phd.correct([array([5.])])

        Args:
            birth_belief: GMM of target births
            survival_rate: Probability of a target to survive a timestep
            detection_rate: Probability of a target to be detected at a timestep
            intensity: Clutter intensity
            F: Linearstate transition model
            H: Linear measurement model
            Q: Process noise matrix
            R: Measurement noise matrix
        """

        # Filter specification
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        # Gaussian mixture model for spontaneous birth of new targets
        self.birth_belief = birth_belief

        # Rates of survival, detection and clutter intensity
        self.survival_rate = survival_rate
        self.detection_rate = detection_rate
        self.intensity = intensity

        # Gaussian mixture model
        self.gmm: List[Gaussian] = []

    def extract(self, threshold: float = 0.5) -> List[ndarray]:
        """Extract a state representation based on spikes in the current GMM.

        Args:
            threshold: Weight that a component needs to have to be considered a target state
        """

        # Memory for all estimated states
        states: List[ndarray] = []

        # Every component with sufficient weight is considered to be a target
        for component in self.gmm:
            if component.w > threshold:
                # A component with weight over 1 represents multiple targets
                states += [component.x for _ in range(int(round(component.w)))]

        # Return all extracted states
        return states

    def prune(self, threshold: ndarray, merge_distance: ndarray, max_components: int):
        """Reduces the number of gaussian mixture components.

        Args:
            threshold: Truncation threshold s.t. components with weight < threshold are removed
            merge_distance: Merging threshold s.t. components 'close enough' will be merged
            max_components: Maximum number of gaussians after pruning
        """

        # Select a subset of components to be pruned
        selected = [component for component in self.gmm if component.w > threshold]

        # Create new list for pruned mixture model
        pruned: List[Gaussian] = []

        # While candidates for pruning exist ...
        while selected:
            # Find mean of component with maximum weight
            index = max(range(len(selected)), key=lambda index: selected[index].w)

            mean = selected[index].x

            # Select components to be merged and remove merged from selected
            mergeable = [
                c for c in selected if ((c.x - mean).T @ inv(c.P) @ (c.x - mean)).item() <= merge_distance
            ]
            selected = [c for c in selected if c not in mergeable]

            # Compute new mixture component
            merged_weight = sum([component.w for component in mergeable])

            merged_mean = sum([component.w * component.x for component in mergeable]) / merged_weight

            merged_covariance = (
                sum(
                    [
                        component.w * (component.P + (mean - component.x) @ (mean - component.x).T)
                        for component in mergeable
                    ]
                )
                / merged_weight
            )

            # Store the component
            pruned.append(Gaussian(merged_mean, merged_covariance, merged_weight))

        # Remove components with minimum weight if maximum number is exceeded
        while len(pruned) > max_components:
            # Find index of component with minimum weight
            index = min(range(len(pruned)), key=lambda index: pruned[index].w)

            # Remove the component
            del pruned[index]

        # Update GMM with pruned model
        self.gmm = deepcopy(pruned)

    def predict(self, **kwargs) -> None:
        """Predict the future state."""

        # Compute F if additional parameters are needed
        if callable(self.F):
            F = self.F(**kwargs)
        else:
            F = self.F

        # Spontaneous birth
        born = deepcopy(self.birth_belief)

        # Spawning off of existing targets
        spawned: List[Gaussian] = []

        # Prediction for existing targets
        # m_k|k-1 = F*m_k-1
        # w_k|k-1 = p_s*w_k-1
        # P_k|k-1 = F*P_k-1*F^T + Q_k-1
        for component in self.gmm:
            component.x = F @ component.x
            component.P = F @ component.P @ F.T + self.Q
            component.w *= self.survival_rate

        # Concatenate with newborn and spawned target components
        self.gmm += born + spawned

    def correct(self, measurements: ndarray, **kwargs) -> None:
        """Correct the former prediction based on a sensor reading.

        Args:
            measurements: Measurements at this timestep
        """

        # Check for differing measurement model
        H = kwargs.pop("H", self.H)

        # Compute H if additional parameters are needed
        if callable(H):
            H = H(**kwargs)

        # ######################################
        # Construction of update components 

        mu: List[ndarray] = []  # Means mapped to measurement space
        S: List[ndarray] = []  # Residual covariance
        K: List[ndarray] = []  # Gains
        P: List[ndarray] = []  # Covariance

        'Fragen!!'
        # Wie ist J_k|k-1 definiert?
        # warum zip ??
        'Fragen!!'
        for i, component in zip(range(len(self.gmm)), self.gmm):
            mu.append(H @ component.x)
            S.append(self.R + H @ component.P @ H.T)
            K.append(component.P @ H.T @ inv(S[i]))
            'Fehler??'
            P.append(component.P - K[i] @ S[i] @ K[i].T)  # [I-K*H]*P != P-K*H*K^T entspricht nicht der Gl. aus Paper Vo et al Nov2006
            #P.append(component.P - K[i] @ S[i] @ component.P)
            'Fehler??'

        # ######################################
        # Update

        # Undetected assumption
        updated = deepcopy(self.gmm)
        for component in updated:
            component.w *= 1 - self.detection_rate # w_k = (1-p_d)*w_k|k-1

        # Measured assumption
        for z in measurements:
            # Create new batch of components for this measurement
            batch = []

            # Fill batch with corrected components
            'Fragen!!'
            # Warum keine laufvariable l nötig?
            # Was mit indizes (l*Jk|k-1+j)
            'Fragen!!'
            for i in range(len(self.gmm)):
                batch.append(
                    Gaussian(
                        self.gmm[i].x + K[i] @ (z - mu[i]),
                        P[i],
                        self.detection_rate * self.gmm[i].w * Gaussian(mu[i], S[i])(z),
                    )
                )

            # Normalize weights
            for component in batch:
                component.w /= self.intensity + sum([c.w for c in batch])

            # Append batch to updated GMM
            updated += batch

        # Set updated as new gaussian mixture model
        self.gmm = updated


