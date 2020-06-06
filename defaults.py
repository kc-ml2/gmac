def atari():
    return dict(
        num_workers=64,
        batch_size=512,
        epoch=4,
        n_step=128,
        lr=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        cliprange=0.1,
    )


def bullet():
    return dict(
        num_workers=64,
        batch_size=2048,
        epoch=10,
        n_step=512,
        lr=1.0e-4,
        ent_coef=0.001,
        vf_coef=0.5,
        cliprange=0.2,
    )


def atari_acktr():
    return dict(
        num_workers=64,
        batch_size=1280,
        epoch=1,
        n_step=20,
        lr=0.25,
        ent_coef=0.01,
        vf_coef=0.5,
    )


def bullet_acktr():
    return dict(
        num_workers=64,
        batch_size=5120,
        epoch=1,
        n_step=80,
        lr=0.1,
        ent_coef=0.001,
        vf_coef=0.5,
    )


def atari_a2c():
    return dict(
        num_workers=64,
        batch_size=320,
        epoch=1,
        n_step=5,
        lr=7e-4,
        ent_coef=0.01,
        vf_coef=0.5,
    )


def bullet_a2c():
    return dict(
        num_workers=64,
        batch_size=2048,
        epoch=1,
        n_step=128,
        lr=7e-4,
        ent_coef=0.001,
        vf_coef=0.5,
    )
