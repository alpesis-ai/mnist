import numpy as np

import theano
from theano.tensor as T


sharedX = (lambda X, name: theano.shared(np.asarray(X, dtype=theano.config.floatX),
                                         name=name))


class HMCSampler(object):
    """
    Convenience wrapper for performing Hybrid Monte Carlo (HMC).

    It creates the symbolic graph for performing an HMC simulation (using
    `hmc_move` and `hmc_updates`). The graph is then compiled into the
    `simulate` function, a theano function which runs the simulation
    and updates the required shared variables.

    Users should interface with the sampler through the `draw` function
    which advances the markov chain and returns the current sample by calling
    `simulate` and `get_position` in sequence.

    The hyper-parameters are the same as those used by Marc'Aurelio''s
    `train_mcRBM.py` file.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


    @classmethod
    def new_from_shared_positions(cls,
                                  shared_positions,
                                  energy_fn,
                                  initial_stepsize=0.01,
                                  target_acceptance_rate=.9,
                                  n_steps=20,
                                  stepsize_dec=0.98,
                                  stepsize_min=0.001,
                                  stepsize_max=0.25,
                                  stepsize_inc=1.02,
                                  avg_acceptance_slowness=0.9,
                                  seed=12345):
        """
        :type shared_positions: theano.ndarray
        :param shared_positions: 

        :type energy_fn: 
        :type energy_fn:

        Outputs:

        :type xx:
        :param xx:
        """

        # allocate shared variables
        stepsize = sharedX(initial_stepsize, 'hmc_stepsize')
        avg_acceptance_rate = sharedX(target_acceptance_rate, 'avg_acceptance_rate')
        s_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)

        # define graph for an `n_steps` HMC simulation
        accept, final_pos = hmc_move(s_rng,
                                     shared_positions,
                                     energy_fn,
                                     stepsize,
                                     n_steps)

        # define the dictionary of updates, to apply on every `simulate` call
        simulate_updates = hmc_updates(shared_positions,
                                       stepsize,
                                       avg_acceptance_rate,
                                       final_pos=final_pos,
                                       accept=accept,
                                       stepsize_min=stepsize_min,
                                       stepsize_max=stepsize_max,
                                       stepsize_inc=stepsize_inc,
                                       stepsize_dec=stepsize_dec,
                                       target_acceptance_rate=target_acceptance_rate,
                                       avg_acceptance_slowness=avg_acceptance_slowness)

        # compile theano function
        simulate = function([], [], updates=simulate_updates)

        # create HMCSampler object with the following attributes
        return cls(positions=shared_positions,
                   stepsize=stepsize,
                   stepsize_min=stepsize_min,
                   stepsize_max=stepsize_max,
                   avg_acceptance_rate=avg_acceptance_rate,
                   target_acceptance_rate=target_acceptance_rate,
                   s_rng=s_rng,
                   _updates=simulate_updates,
                   simulate=simulate)


    def draw(self, **kwargs):
        """
        Returns a new position obtained after `n_steps` of HMC simulation.

        :type kwargs: dictionary
        :param kwargs:

        Outputs:

        :type rval: numpy.matrix
        :param rval:
        """

        self.simulate()
        return self.positions.get_value(borrow=False) 


def hamiltonian(pos, vel, energy_fn):
    """ Returns the Hamiltonian (sum of potential and kinetic energy) for the
    given velocity and position.

    :type pos: theano.matrix
    :param pos: symbolic matrix whose rows are position vectors

    :type vel: theano.matrix
    :param vel: symbolic matrix whose rows are velocity vectors

    :type energy_fn: python function
    :param energy_fn: operating on symbolic theano variables, used tox compute
                      the potential energy at a given position

    :returns: theano.vector, vector whose i-th entry is the Hamiltonian at
              position pos[i] and velocity vel[i]
    """
    # assuming mass is 1
    return energy_fn(pos) + kinetic_energy(vel)


def kinetic_energy(vel):
    """ Returns the kinetic energy associated with the given velocity
    and mass of 1.

    :type vel: theano.matrix
    :param vel: symbolic matrix whose rows are velocity vectors

    :return: theano.vector, whose i-th entry is the kinetic entry
             associated with vel[i]
    """
    return 0.5 * (vel ** 2).sum(axis=1)


def metropolis_hastings_accept(energy_prev, energy_next, s_rng):
    """
    Performs a Metropolis-Hastings accept-reject move.

    :type energy_prev: theano.vector
    :param energy_prev: symbolic theano tensor which contains the energy associated
                        with the configuration at time-step t

    :type energy_next: theano.vector
    :param energy_next: symbolic theano tensor which contains the energy associated
                        with the proposed configuration at time-step t+1

    :type s_rng: theano.tensor.shared_randomstreams.RandomStreams
    :param s_rng: theano shared random stream object used to generate the random
                  number used in proposal

    :returns: boolean, True if move is accepted, False otherwise
    """
    ediff = energy_prev - energy_next
    return (T.exp(ediff) - s_rng.uniform(size=energy_prev.shape)) >= 0


def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
    """
    Returns final (position, velocity) obtained after an `n_steps` leapfrog
    updates, using Hamiltonian dynamics.

    :type initial_pos: shared theano.matrix
    :param initial_pos: initial position at which to start the simulation

    :type initial_vel: shared theano.matrix
    :param initial_vel: initial velocity of particles

    :type stepsize: shared theano scalar
    :param stepsize: scalar value controlling amount by which to move

    :type n_steps: 
    :param n_steps:

    :type energy_fn: python function
    :param energy_fn: python function, operating on symbolic theano variables,
                      used to compute the potential energy at a given position

    Outputs:

    :type rval1: theano.matrix
    :param rval1: final positions obtained after simulation
    
    :type rval2: theano.matrix
    :param rval2: final velocity obtained after simulation
    """

    def leapfrog(pos, vel, step):
        """
        Inside loop of scan, performs one step of leapfrog update, using
        Hamiltonian dynamics.

        :type pos: theano.matrix
        :param pos: in leapfrog update equations, represents pos(t), position
                    at time t

        :type vel: theano.matrix
        :param vel: in leapfrog update equations, represents vel(t - stepsize/2),
                    velocity at time (t - stepsize/2)

        :type step: theano.scalar
        :param step: scalar value controlling amount by which to move

        Outputs:
        
        :type rval1: [theano.matrix, theano.matrix]
        :param rval1: symbolic theano matrics for new position pos(t + stepsize),
                      and velocity vel(t + stepsize/2)

        :type rval2: dictionary
        :param rval2: dictionary of updates for the scan op
        """

        # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
        dE_dpos = T.grad(energy_fn(pos).sum(), pos)
        new_vel = vel - step * dE_dpos

        # from vel(t+stepsize//2) compute pos(t+stepsize)
        new_pos = pos + step * new_vel
  
        return [new_pos, new_vel], {}


    # compute velocity at time-step: t + stepsize // 2
    initial_energy = energy_fn(initial_pos)
    dE_dpos = T.grad(initial_energy.sum(), initial_pos)
    vel_half_step = intial_vel - 0.5 * stepsize * dE_dpos

    # compute position at time-step: t + stepsize
    pos_full_step = initial_pos + stepsize * vel_half_step

    # perform leapfrog updates: the scan op is used to repeatedly compute
    # vel(t + (m-1/2)*stepsize) and pos(t + m*stepsize) for m in [2, n_steps]
    outputs_info = [
        dict(initial=pos_full_step),
        dict(initial=vel_half_step),
    ]
    (all_pos, all_vel), scan_updates = theano.scan(leapfrog,
                                                   outputs_info=outputs_info,
                                                   non_sequences=[stepsize],
                                                   n_steps=n_steps - 1)
    final_pos = all_pos[-1]
    final_vel = all_vel[-1]

    # NOTE: scan always returns an updates dictionary, in case the scanned
    # function draws samples from a RandomStream. 
    # These updates must then be used when compiling the Theano function,
    # to avoid drawing the same random numbers each time the function is
    # called.
    # In this case, however, we consiciously ignore `scan_updates` because
    # we know it is empty
    assert not scan_updates

    # the last velocity returned by scan is vel(t + (n_steps - 1/2) * stepsize)
    # We therefore perform one more half-step to return vel(t + n_steps * stepsize)
    energy = energy_fn(final_pos)
    final_vel = final_vel - 0.5 * stepsize * T.grad(energy.sum(), final_pos)
  
    # return new proposal state
    return final_pos, final_vel


def hmc_move(s_rng, positions, energy_fn, stepsize, n_steps):
    """
    This function performs one-step of Hybrid Monte-Carlo sampling. We start
    by sampling a random velocity from a univariate Gaussian distribution,
    perform `n_steps` leap-frog updates using Hamiltonian dynamics and
    accept-reject using Metropolis-Hastings.

    :type s_rng: theano.shared.random_stream
    :param s_rng: symbolic random number generator used to draw random
                  velocity and perform accept-reject move

    :type positions: (shared) theano.matrix
    :param positions: symbolic matrix whose rows are position vectors

    :type energy_fn: python function
    :param energy_fn: operating on symbolic theano variables, used to
                      compute the potential energy at a given position

    :type stepsize: (shared) theano.scalar
    :param stepsize: shared variable containing the stepsize to use for
                     `n_steps` of HMC simulation steps

    :type n_steps: integer
    :param n_steps: number of HMC steps to perform before proposing a
                    new position

    Outputs:

    :type rval1: boolean
    :param rval1: True if move is accepted, False otherwise

    :type rval2: theano.matrix
    :param rval2: matrix whose rows contain the proposed `new position`
    """

    # sample random velocity
    initial_vel = s_rng.normal(size=positions.shape)
    # perform simulation of particles subject to Hamiltonian dynamics
    final_pos, final_vel = simulate_dynamics(initial_pos=positions,
                                             initial_vel=initial_vel,
                                             stepsize=stepsize,
                                             n_steps=n_steps,
                                             energy_fn=energy_fn)

    # accept/reject the proposed move based on the joint distribution
    accept = metropolis_hastings_accept(energy_prev=hamiltonian(positions, initial_vel, energy_fn),
                                        energy_next=hamiltonian(final_pos, final_vel, energy_fn),
                                        s_rng=s_rng)

    return accept, final_pos


def hmc_updates(positions,
                stepsize,
                avg_acceptance_rate,
                final_pos,
                accept,
                target_acceptance_rate,
                stepsize_inc,
                stepsize_dec,
                stepsize_min,
                stepsize_max,
                avg_acceptance_slowness):
    """
    This function is executed after `n_steps` of HMC sampling (`hmc_move` function).
    It creates the updates dictionary used by the `simulate` function.
    It takes care of updating:
    - the position (if the move is accepted)
    - the stepsize (to track a given target aceptance rate)
    - the average acceptance rate (computed as a moving average)

    Inputs:

    :type positions: (shared) theano.matrix
    :param positions: shared theano matrix whose rows contain the old position

    :type stepsize: (shared) theano.scalar
    :param stepsize: shared theano scalar containig current step size

    :type avg_acceptance_rate: (shared) theano.scalar
    :param avg_acceptance_rate: the current average acceptance rate

    :type final_pos: (shared) theano.matrix
    :param final_pos: shared theano matrix whose rows contain the new position

    :type accept: theano.scalar
    :param accept: boolean-type variable representing whether or not the
                   proposed HMC move should be accepted or not

    :type target_acceptance_rate: float
    :param target_acceptance_rate: the stepsize is modified in order to track
                                   this target acceptance rate

    :type stepsize_inc: float
    :param stepsize_inc: amount by which to increment stepsize when acceptance
                         rate is too high

    :type stepsize_dec: float
    :param stepsize_dec: amount by which to decrement stepsize when acceptance
                         rate is too low

    :type stepsize_min: float
    :param stepsize_min: lower-bound on `stepsize`

    :type stepsize_max: float
    :param stepsize_max: upper-bound on `stepsize`

    :type avg_acceptance_slowness: float
    :param avg_acceptance_slowness: average acceptance rate is computed as an
                                    exponential moving average.

    Outputs:

    :type rval1: dictionary-like
    :param rval1: a dictionary of updates to be used by the `HMCSampler.simulate`
                  function. The updates target the position, stepsize and average
                  acceptance rate
    """

    # Position updates
    # broadcast `accept` scalar to tensor with the same dimensions as final_pos
    accept_matrix = accept.dimshuffle(0, *(('x',) * (final_pos.ndim -1)))
    # if accept is True, update to `final_pos` else stay put
    new_positions = T.switch(accept_matrix, final_pos, positions)

    # Stepsize updates
    # if acceptance rate is too low, our sampler is too "noisy" and we reduce
    # the stepsize. If it is too high, our sampler is too conservative, we can
    # get away with a larger stepsize (resultingn in better mixing)
    _new_stepsize = T.switch(avg_acceptance_rate > target_acceptance_rate,
                             stepsize * stepsize_inc,
                             stepsize * stepsize_dec)
    # maintain stepsize in [stepsize_min, stepsize_max]
    new_stepsize = T.clip(_new_stepsize, stepsize_min, stepsize_max)

    # Accept rate updates
    # perform exponential moving average
    mean_dtype = theano.scalar.upcast(accept.dtype, avg_acceptance_rate.dtype)
    new_acceptance_rate = T.add(avg_acceptance_slowness * avg_acceptance_rate,
                                (1.0 - avg_acceptance_slowness) * accept.mean(dtype=mean_dtype))
    return [
        (positions, new_positions),
        (stepsize, new_stepsize),
        (avg_acceptance_rate, new_acceptance_rate)
    ]
