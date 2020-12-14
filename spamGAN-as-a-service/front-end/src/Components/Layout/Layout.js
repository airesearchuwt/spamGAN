import React from 'react';
import Aux from '../../hoc/Aux';
import ParticlesBg  from "particles-bg";
import Navigation from '../Navigation/Navigation';
import Main from '../Main/Main';

const Layout = (props) => (
    <Aux>
        <ParticlesBg type="circle" bg={true} />
        <Navigation />
        <Main />
    </Aux>
);

export default Layout;