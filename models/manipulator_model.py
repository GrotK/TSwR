import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=0.01, r3=0.01):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 3.
        self.l2 = 0.4
        self.r2 = 0.01
        self.m2 = 2.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        # Dlugosci do srodkow mas:
        d1 = self.l1 / 2
        d2 = self.l2 / 2

        # Zmienne pomocnicze
        alpha = self.m1 * d1 ** 2 + self.I_1 + self.m2 * self.l1 ** 2 + self.m2 * d2 ** 2 + self.I_2 + self.m3 * self.l1 ** 2 + self.m3 * self.l2 ** 2 + self.I_3
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        gamma = self.m2 * d2 ** 2 + self.m3 * self.l2 ** 2 + self.I_2 + self.I_3

        # Wspolczynniki macierzy M(q)
        m_11 = alpha + 2 * beta * np.cos(q2)
        m_12 = gamma + beta * np.cos(q2)
        m_21 = gamma + beta * np.cos(q2)
        m_22 = gamma

        return np.array([[m_11, m_12], [m_21, m_22]])

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        d2 = self.l2 / 2
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2

        # Wspolczynniki macierzy C(q,q')
        c_11 = -beta * np.sin(q2) * q2_dot
        c_12 = -beta * np.sin(q2) * (q1_dot + q2_dot)
        c_21 = beta * np.sin(q2) * q1_dot
        c_22 = 0
        return np.array([[c_11, c_12], [c_21, c_22]])

    # Calculate x_dot
    def x_dot(self, x, u):
        invM = np.linalg.inv(self.M(x))
        zeros = np.zeros((2, 2), dtype=np.float32)
        # A = -inv(M)*C
        A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1), np.concatenate([zeros, -invM @ self.C(x)], 1)], 0)
        # b = inv(M)
        b = np.concatenate([zeros, invM], 0)
        return A @ x[:, np.newaxis] + b @ u